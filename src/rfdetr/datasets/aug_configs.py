# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Optional Albumentations augmentation presets for RF-DETR training.

RF-DETR's default training augmentation path is torchvision-native and does not require Albumentations. Importing and
passing the presets in this module as ``aug_config`` uses the optional Albumentations integration; install it with
``pip install 'rfdetr[augment]'``.

Import a preset and pass it as ``aug_config`` to your training call:

```python
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

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
| ``AUG_MOTION_RGB``    | RGB + motion channel, layout ``[R, G, B, flow]`` |
| ``AUG_MOTION_GREY``   | Motion + grey, layout ``[motion, grey, motion]`` |
| ``AUG_MOTION_ONLY``   | All-motion input, three motion channels       |

The ``AUG_MOTION_*`` presets are lists (they repeat the ``ChannelSubset`` container) and need the Albumentations
backend.

## Transform Categories

**Geometric transforms** (automatically transform bounding boxes; this representative list is not exhaustive):
- Flips and square symmetries: HorizontalFlip, TimeReverse, VerticalFlip, D4, SquareSymmetry
- Rotations: Rotate, Affine, ShiftScaleRotate
- Crops: RandomCrop, CenterCrop, RandomResizedCrop
- Perspective: Perspective, ElasticTransform, GridDistortion

**Pixel-level transforms** (preserve bounding boxes):
- Color: ColorJitter, HueSaturationValue, RandomBrightnessContrast, ToGray
- Blur/Noise: GaussianBlur, GaussNoise, Blur
- Enhancement: CLAHE, Sharpen, Equalize

## Best Practices

1. **Start conservative**: Use moderate probabilities (p=0.3-0.5) and small parameter ranges
2. **Geometric caution**: Extreme rotations (>45°) or crops may remove too many boxes
3. **Keypoint safety**: Keypoint pipelines with an empty `keypoint_flip_pairs` list disable
   all horizontal-flip-capable aliases, including `TimeReverse`, `D4`, and `SquareSymmetry`;
   provide left/right pairs to retain them.
4. **Performance**: Fewer transforms = faster training; prioritize transforms that match your domain
5. **Validation**: Monitor validation mAP - excessive augmentation can hurt performance
6. **Domain-specific**: Enable augmentations that reflect real-world variations in your data

## Adding Custom Transforms

For geometric transforms not in GEOMETRIC_TRANSFORMS set, add them in transforms.py:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",  # Add here
}
```

## Kornia GPU Backend

When ``augmentation_backend="kornia"`` is set in ``TrainConfig`` (or ``"auto"``/``"cpu"`` resolves to it because
Kornia is installed and CUDA is available), augmentations run on the GPU via Kornia instead of CPU Albumentations
or torchvision defaults. Install it with ``pip install 'rfdetr[augment]'``.

**Supported transforms** (all presets):

| Preset key | Kornia equivalent | Notes |
|---|---|---|
| ``HorizontalFlip`` | ``K.RandomHorizontalFlip`` | Direct |
| ``VerticalFlip`` | ``K.RandomVerticalFlip`` | Direct |
| ``Rotate`` | ``K.RandomRotation`` | ``limit`` may be scalar or tuple |
| ``Affine`` | ``K.RandomAffine`` / pixel translation | Pixel shifts or fractional translation; see below |
| ``ColorJitter`` | ``K.ColorJiggle`` | Same multiplicative semantics |
| ``ToGray`` | ``K.RandomGrayscale`` | Grayscale, 3 channels; only ``p`` honored, method/num_output_channels ignored |
| ``RandomBrightnessContrast`` | ``K.ColorJiggle`` | ``brightness_limit`` / ``contrast_limit`` direct |
| ``GaussianBlur`` | ``K.RandomGaussianBlur`` | ``blur_limit`` rounded up to odd; ``sigma`` mirrors Albumentations |
| ``GaussNoise`` | ``K.RandomGaussianNoise`` | Upper bound of ``std_range`` used as fixed std |
| ``Blur`` | ``K.RandomBoxBlur`` | Box blur; ``blur_limit`` rounded up to odd, pair collapses to its upper bound |
| ``Sharpen`` | ``K.RandomSharpness`` | ``sharpness = 1.0 + alpha`` (1.0-pivoted); ``lightness``/``method`` ignored |
| ``Equalize`` | ``K.RandomEqualize`` | Only ``p`` honored; ``mode``/``by_channels``/``mask`` ignored |
| ``CLAHE`` | ``K.RandomClahe`` | Scalar ``clip_limit=v`` -> ``(1, v)``; pairs and ``tile_grid_size`` pass through |
| ``Perspective`` | ``K.RandomPerspective`` | Approximate; see the note below. ``keep_size=False`` raises |
| ``ShiftScaleRotate`` | ``K.RandomAffine`` | Limits are deltas, not absolute ranges; see the note below |

``Perspective`` is the one entry in the table that is not a faithful mapping. Albumentations samples each
corner offset from ``abs(N(0, scale))``, Kornia samples uniformly from ``distortion_scale``, so the two
produce different distortion distributions for the same config. The upper bound of ``scale`` is used as
``distortion_scale`` and a scalar ``scale`` is read as ``(0, scale)``; the divergence is logged for every
config, not only for a range. ``keep_size=False`` raises rather than silently resizing, and
``fit_output``, ``interpolation``,
``mask_interpolation``, ``border_mode``, ``fill`` and ``fill_mask`` are ignored. Use the Albumentations
backend when the exact Albumentations semantics matter.

``ShiftScaleRotate`` maps onto the same ``K.RandomAffine`` as ``Affine``, which is what makes it safe here:
it preserves the output resolution, so the padding-mask constraint that keeps the crops unsupported does not
apply. Its limits are deltas rather than absolute ranges, so they are converted rather than forwarded.
``scale_limit`` is biased by 1 in Albumentations, meaning it samples from ``(1 + low, 1 + high)``, so the
default ``(-0.1, 0.1)`` is a scale between ``0.9`` and ``1.1``; that pivot is applied here. All three limits
also read a scalar ``v`` as the symmetric ``(-v, v)``. ``shift_limit_x`` and ``shift_limit_y`` map onto
Kornia's per-axis ``translate``; ``None`` uses the shared ``shift_limit``. Kornia can sample only symmetric shift
ranges, so an asymmetric Albumentations pair such as ``(0.1, 0.2)`` becomes ``(-0.2, 0.2)`` and emits a warning.
Use the Albumentations backend when that distribution must be preserved. ``interpolation``, ``border_mode``,
``mask_interpolation``, ``fill``, ``fill_mask`` and ``rotate_method`` have no equivalent and are ignored with a warning.
Note that
Albumentations itself deprecates ``ShiftScaleRotate`` in favour of ``Affine``, which this backend already
supports; new configs should prefer ``Affine``.

``Affine(translate_px=...)`` accepts fixed offsets, inclusive ``(min, max)`` pairs, and per-axis ``x``/``y``
mappings on Kornia; an offset may be any integer or an integral float such as ``4.0``. It applies whole-pixel
shifts to images, boxes, instance masks and padding masks together. Options that cannot change a whole-pixel
shift are accepted and have no effect: ``interpolation`` and ``mask_interpolation`` codes 0-4, ``keep_ratio``,
``balanced_scale``, ``rotate_method``, a unit ``scale`` or zero ``shear``, and ``fill``/``fill_mask`` with a
replicate border. Pipeline construction raises for nonzero rotation or shear, nonunit scale (a per-axis
``scale``/``shear`` mapping's missing axis counts as ``1``, as on Albumentations, so ``shear={"x": 0}`` raises),
``fit_output=True``, interpolation codes outside 0-4, border modes other than constant or replicate, a nonzero
``fill``/``fill_mask`` with a constant border, the deprecated Albumentations 1.x aliases ``cval``, ``cval_mask``
and ``mode`` (use ``fill``, ``fill_mask`` and ``border_mode``), and boolean, fractional, reversed or
out-of-64-bit-range offsets. Use Albumentations for those combinations.
The existing ``translate_percent`` mapping uses a size fraction; a scalar becomes a signed Kornia range instead
of Albumentations' fixed positive shift and emits a warning. A scalar ``scale`` becomes ``(v, v)``.

Not yet supported on Kornia: ``HueSaturationValue`` (Albumentations shifts hue/saturation/value additively,
Kornia's ``ColorJiggle`` scales them multiplicatively, so there is no faithful mapping), and the geometric
group ``RandomCrop``, ``CenterCrop``, ``RandomResizedCrop``,
``ElasticTransform`` and ``GridDistortion``, whose CPU behavior is not faithfully mapped by the current
Kornia pipeline. These still work on the Albumentations backend.

The GPU augmentation path transports the padded-batch mask with every geometric transform. Segmentation models also
carry instance-mask channels in the same synchronized mask tensor.
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

#: --------------------------------------------------------------------------
#: Building blocks. Every op below is probabilistic: nothing fires on every frame,
#: so the model always sees a mixture of clean and perturbed inputs.
#: --------------------------------------------------------------------------

#: Additive/nonlinear intensity ops. Safe ONLY on appearance channels -- on a motion
#: channel these lift stationary background off true zero and invent motion.
_APPEARANCE_INTENSITY = [
    {"RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}},
    {"RandomGamma": {"gamma_limit": (80, 120), "p": 0.3}},
]

#: Occlusion of the appearance channels only. The motion channels still show the
#: fish, so this is the mirror image of motion dropout: it stops the model leaning
#: entirely on appearance, exactly as motion dropout stops it leaning on motion.
#: Deliberately applied inside ChannelSubset, so bounding boxes are untouched -- the
#: target is still there, just hidden in this modality.
_APPEARANCE_OCCLUSION = [
    {
        "CoarseDropout": {
            "num_holes_range": (1, 4),
            "hole_height_range": (0.05, 0.15),
            "hole_width_range": (0.05, 0.15),
            "fill": 0,
            "p": 0.2,
        }
    },
]

#: Underwater domain shift: backscatter/turbidity and defocus. RandomFog requires a
#: 3-channel input, so it is only valid on an RGB subset -- on a single grey channel
#: it raises and would be silently skipped by the builder.
_APPEARANCE_TURBIDITY_RGB = [
    {"RandomFog": {"fog_coef_range": (0.1, 0.35), "alpha_coef": 0.08, "p": 0.15}},
    {"GaussianBlur": {"blur_limit": (3, 5), "p": 0.15}},
]
_APPEARANCE_TURBIDITY_GREY = [
    {"GaussianBlur": {"blur_limit": (3, 5), "p": 0.15}},
]

#: Gain jitter for motion channels: one random factor per image, clipped at the
#: dtype ceiling. Multiplication is the one intensity op that is safe on a motion
#: channel, because 0 * k == 0 leaves stationary background exactly zero. It buys
#: invariance to the absolute motion scale, which is an artifact of the optical-flow
#: scale parameter, the byte scale_factors and how fast the fish happened to be
#: swimming -- not a property of the fish.
_MOTION_GAIN = {
    "MultiplicativeNoise": {"multiplier": (0.7, 1.3), "per_channel": False, "elementwise": False, "p": 0.5}
}

#: Per-pixel multiplicative jitter: flow and variance estimates are genuinely noisy,
#: and being multiplicative this also preserves the zero background exactly.
_MOTION_NOISE = {
    "MultiplicativeNoise": {"multiplier": (0.8, 1.2), "per_channel": False, "elementwise": True, "p": 0.3}
}


def _motion_blank(p: float) -> dict:
    """Blank the channel entirely, on ``p`` of frames (multiply by zero).

    This is modality dropout, and it targets a failure that provably occurs:
    ocv_optical_flow emits an all-zero field on its first frame and after any
    resolution change, and a scene cut has the same effect. A model that has learned
    to lean on the motion channel produces garbage on exactly those frames. Firing
    this on a slice of frames forces it to degrade gracefully to appearance instead.

    Inside a OneOf, ``p`` is a selection weight rather than a firing probability, so
    the two call sites pass different values.
    """
    return {
        "MultiplicativeNoise": {
            "multiplier": (0.0, 0.0),
            "per_channel": False,
            "elementwise": False,
            "p": p,
        }
    }


#: Geometric ops. Safe on every channel: they move pixels without reinterpreting
#: their values. No VerticalFlip -- up is up in this footage, and flipping it
#: manufactures upside-down subjects that never occur at inference.
_MOTION_GEOMETRIC = [
    {"HorizontalFlip": {"p": 0.5}},
    {"Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "rotate": (-10, 10), "p": 0.5}},
]

# Presets are lists, not dicts: they carry more than one ChannelSubset entry (one per
# group of channels) and a dict keyed by transform name cannot repeat a key.

#: RGB + optical-flow magnitude, layout [ R, G, B, flow ].
AUG_MOTION_RGB = [
    *_MOTION_GEOMETRIC,
    {
        "ChannelSubset": {
            "channels": [0, 1, 2],
            "transforms": _APPEARANCE_INTENSITY
            + [{"ColorJitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.2, "hue": 0.05, "p": 0.3}}]
            + _APPEARANCE_TURBIDITY_RGB
            + _APPEARANCE_OCCLUSION,
            "p": 1.0,
        }
    },
    {
        "ChannelSubset": {
            "channels": [3],
            # 10% of frames lose the flow channel entirely and must fall back to RGB.
            "transforms": [_MOTION_GAIN, _MOTION_NOISE, _motion_blank(0.10)],
            "p": 1.0,
        }
    },
]

#: Two motion channels around a greyscale one, layout [ motion, grey, motion ].
#: No ColorJitter or RandomFog: both need 3 channels, and the appearance subset is 1.
AUG_MOTION_GREY = [
    *_MOTION_GEOMETRIC,
    {
        "ChannelSubset": {
            "channels": [1],
            "transforms": _APPEARANCE_INTENSITY + _APPEARANCE_TURBIDITY_GREY + _APPEARANCE_OCCLUSION,
            "p": 1.0,
        }
    },
    {
        "ChannelSubset": {
            "channels": [0, 2],
            # Both variance channels are the same modality, so they blank together:
            # 10% of frames fall back to the grey channel alone.
            "transforms": [
                _MOTION_GAIN,
                _MOTION_NOISE,
                _motion_blank(0.10),
            ],
            "p": 1.0,
        }
    },
]

#: All-motion input, layout [ flow, frame_difference, temporal_variance ]. Every
#: channel is a motion channel, so gain and noise apply to the whole image.
#:
#: Dropout is different here: blanking every channel would leave an empty image that
#: still carries boxes, which is label noise, not augmentation. The OneOf picks
#: exactly ONE channel to blank (or, 85% of the time, none), so the model learns to
#: cope when any single operator fails but never sees a blank frame.
AUG_MOTION_ONLY = [
    *_MOTION_GEOMETRIC,
    _MOTION_GAIN,
    _MOTION_NOISE,
    {
        "OneOf": {
            "transforms": [
                {"ChannelSubset": {"channels": [0], "transforms": [_motion_blank(1.0)], "p": 0.05}},
                {"ChannelSubset": {"channels": [1], "transforms": [_motion_blank(1.0)], "p": 0.05}},
                {"ChannelSubset": {"channels": [2], "transforms": [_motion_blank(1.0)], "p": 0.05}},
                {"NoOp": {"p": 0.85}},
            ]
        }
    },
]
