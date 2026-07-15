# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.shapes import as_pair

logger = get_logger()


def is_valid_coco_dataset(dataset_dir: str) -> bool:
    return (Path(dataset_dir) / "train" / "_annotations.coco.json").exists()


def compute_multi_scale_scales(
    resolution: int | Tuple[int, int],
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> List[Tuple[int, int]]:
    """Derive the multi-scale jitter ladder around *resolution*.

    Scales are stepped in whole blocks of ``patch_size * num_windows`` so every candidate remains patchable and
    windowable.  For a non-square resolution the aspect ratio is held fixed: the shorter side is stepped in whole
    blocks and the longer side is scaled proportionally, then snapped back to a whole number of blocks.

    Args:
        resolution: Base resolution, either a side length (square) or ``(height, width)``.
        expanded_scales: Widen the jitter range.
        patch_size: Model patch size.
        num_windows: Windows per spatial dimension.

    Returns:
        Candidate ``(height, width)`` resolutions, each with both dims divisible by
        ``patch_size * num_windows``.
    """
    block = patch_size * num_windows
    height, width = as_pair(resolution)

    # Step the shorter side in whole blocks and carry the longer side along at a fixed aspect.
    short, long = (height, width) if height <= width else (width, height)
    aspect = long / short
    base_blocks = short // block

    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    proposed: List[Tuple[int, int]] = []
    for offset in offsets:
        short_blocks = base_blocks + offset
        if short_blocks < 2:  # ensure minimum image size
            continue
        long_blocks = max(2, round(short_blocks * aspect))
        short_px, long_px = short_blocks * block, long_blocks * block
        proposed.append((short_px, long_px) if height <= width else (long_px, short_px))

    return proposed


def _is_rle(segmentation: Any) -> bool:
    """Check whether a COCO segmentation entry is in RLE format.

    RLE annotations are dicts with ``"counts"`` and ``"size"`` keys, as opposed to polygon annotations which are lists
    of coordinate arrays. This is a structural check only — it verifies key presence but does not validate value types.
    A dict with counts=None will pass this check but fail downstream in convert_coco_poly_to_mask.

    Args:
        segmentation: A single COCO segmentation annotation entry.

    Returns:
        ``True`` if the entry looks like an RLE dict, ``False`` otherwise.
    """
    return isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation


def convert_coco_poly_to_mask(segmentations: List[Any], height: int, width: int) -> torch.Tensor:
    """Convert COCO segmentation annotations to a binary mask tensor of shape ``[N, H, W]``.

    Supports both polygon and RLE (Run-Length Encoding) annotation formats. Polygon annotations (lists of coordinate
    arrays) are rasterised via ``pycocotools.mask.frPyObjects``.  RLE annotations (dicts with ``"counts"`` and
    ``"size"`` keys; ``counts`` may be str or bytes for compressed RLE, or list of ints for uncompressed RLE) are
    decoded directly, skipping the polygon-to-RLE conversion step.

    Args:
        segmentations: Per-instance segmentation annotations.  Each element is
            either a polygon list (``[[x1, y1, x2, y2, ...], ...]``), an RLE dict (``{"counts": ..., "size": [H, W]}``),
            or ``None`` / empty for instances without a mask. Dicts must be valid COCO RLE annotations with non-empty
            ``"counts"`` and ``"size"`` fields.
        height: Image height in pixels (used for polygon rasterisation).
        width: Image width in pixels (used for polygon rasterisation).

    Returns:
        A ``uint8`` tensor of shape ``(N, H, W)`` where each slice is a binary mask for one instance.  Returns a ``(0,
        H, W)`` tensor when *segmentations* is empty.
    """
    import pycocotools.mask as coco_mask

    masks = []
    for segmentation in segmentations:
        if segmentation is None or (not isinstance(segmentation, dict) and len(segmentation) == 0):
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        if _is_rle(segmentation):
            counts = segmentation["counts"]
            if not isinstance(counts, (str, bytes, list)):
                raise ValueError(
                    f"RLE segmentation has unsupported counts type {type(counts).__name__!r}; "
                    "expected str, bytes, or list"
                )
            if isinstance(counts, (str, bytes)):
                # Compressed RLE — decode directly, skip frPyObjects
                rles = [segmentation]
            else:
                # Uncompressed RLE (counts is a list of ints) — compress first
                rles = [coco_mask.frPyObjects(segmentation, height, width)]
        else:
            rles = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        # Keep return dtype stable across torch versions (any(...) may return bool).
        mask = mask.any(dim=2).to(torch.uint8)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO detection dataset with optional sparse-to-contiguous category ID remapping.

    Extends ``torchvision.datasets.CocoDetection`` with two additions:

    1. A pluggable transform pipeline (``transforms``) applied after the raw
       annotation conversion handled by :class:`ConvertCoco`.
    2. Optional remapping of sparse COCO category IDs to contiguous 0-based label
       indices via ``remap_category_ids``.

    COCO category IDs are sparse (1–90 with gaps such as 12, 26, 29 …).  When a model has only *N* output slots the IDs
    cannot be used directly as tensor indices — doing so causes out-of-bounds errors in the matcher and loss. Setting
    ``remap_category_ids=True`` builds a ``cat2label`` mapping from the annotation file so that IDs are remapped to the
    range ``[0, N)``.  The reverse ``label2cat`` mapping is attached to the underlying COCO API object so that
    :class:`~rfdetr.datasets.coco_eval.CocoEvaluator` can convert predicted label indices back to the original category
    IDs required by pycocotools.

    ``remap_category_ids`` should be ``True`` for Roboflow / custom datasets (via :func:`build_roboflow_from_coco`) and
    ``False`` (the default) when evaluating pretrained models that were trained with the convention that model output
    slot *k* corresponds directly to COCO category ID *k*.

    Args:
        img_folder: Path to the directory containing the dataset images.
        ann_file: Path to the COCO-format JSON annotation file.
        transforms: Transform pipeline applied to ``(image, target)`` pairs after
            annotation conversion.  ``None`` means no additional transforms.
        include_masks: If ``True``, decode polygon segmentation masks into binary
            tensors and include them in the target dict under the ``"masks"`` key.
        remap_category_ids: If ``True``, build a ``cat2label`` mapping from the
            annotation file that remaps sparse category IDs to contiguous 0-based label indices.  The reverse mapping is
            stored as ``label2cat`` on both this object and the underlying COCO API object.  Defaults to ``False``.
    """

    def __init__(
        self,
        img_folder: Union[str, Path],
        ann_file: Union[str, Path],
        transforms: Optional[Any],
        include_masks: bool = False,
        remap_category_ids: bool = False,
        include_keypoints: bool = False,
        num_keypoints: int = 2,
    ) -> None:
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.include_keypoints = include_keypoints
        self.num_keypoints = num_keypoints
        if remap_category_ids:
            # Mapping from original COCO category_id to contiguous label indices
            self.cat2label = {cat_id: i for i, cat_id in enumerate(sorted(self.coco.cats.keys()))}
            # Reverse mapping from contiguous label indices back to COCO category_id
            self.label2cat = {label: cat_id for cat_id, label in self.cat2label.items()}
            # Expose label-to-category mapping on the underlying COCO API object for evaluators
            setattr(self.coco, "label2cat", self.label2cat)
        else:
            self.cat2label = None
            self.label2cat = None
        self.prepare = ConvertCoco(
            include_masks=include_masks,
            cat2label=self.cat2label,
            include_keypoints=include_keypoints,
            num_keypoints=num_keypoints,
        )

    def _load_image(self, id: int) -> Image.Image:
        # Override torchvision's RGB-forcing loader so imagery with an appended
        # motion/flow channel (saved as 4-channel RGBA) keeps all its channels.
        # Standard 3-channel imagery is still returned as RGB.
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(str(Path(self.root) / path))
        if image.mode in ("RGBA", "LA", "CMYK") or (
            image.mode == "P" and "transparency" in image.info
        ):
            return image.convert("RGBA")
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            # boxes are absolute [x_min, y_min, x_max, y_max]; conversion to
            # normalized [cx, cy, w, h] occurs inside Normalize
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):
    """Convert a raw COCO annotation dict into model-ready tensors.

    Accepts the ``(image, target)`` pair produced by ``torchvision.datasets.CocoDetection`` and returns the same image
    alongside a target dict containing:

    - ``"boxes"`` – ``(N, 4)`` float32 tensor in absolute ``[x_min, y_min, x_max, y_max]`` format.
    - ``"labels"`` – ``(N,)`` int64 tensor of class indices.
    - ``"image_id"`` – scalar int64 tensor.
    - ``"area"`` – ``(N,)`` float32 tensor of annotation areas (used by COCO eval).
    - ``"iscrowd"`` – ``(N,)`` int64 tensor (0 = instance, 1 = crowd).
    - ``"masks"`` – ``(N, H, W)`` bool tensor of binary segmentation masks, only
      present when ``include_masks=True``.

    Crowd annotations (``iscrowd=1``) and degenerate boxes (zero width or height after clamping to image boundaries) are
    filtered out.

    Args:
        include_masks: If ``True``, decode segmentation annotations (polygon or
            RLE format) into binary masks and include them in the returned target dict.
        cat2label: Optional mapping from COCO ``category_id`` values to contiguous
            0-based label indices.  When ``None`` (default) the raw ``category_id`` values are used as labels directly,
            which is correct for datasets whose IDs are already 0-indexed.  Pass a non-``None`` mapping for sparse
            COCO-style datasets (e.g. IDs 1–90 with gaps) so that labels stay within the model's output range.
    """

    def __init__(
        self,
        include_masks: bool = False,
        cat2label: Optional[Dict[int, int]] = None,
        include_keypoints: bool = False,
        num_keypoints: int = 2,
    ) -> None:
        self.include_masks = include_masks
        self.cat2label = cat2label
        self.include_keypoints = include_keypoints
        self.num_keypoints = num_keypoints

    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes: List[int] = []
        for obj in anno:
            category_id = obj["category_id"]
            if getattr(self, "cat2label", None) is not None:
                if category_id not in self.cat2label:
                    raise KeyError(
                        f"Unknown category_id {category_id} for image_id {target.get('image_id')} "
                        "encountered in annotations. Check that your category mapping matches the dataset."
                    )
                classes.append(self.cat2label[category_id])
            else:
                classes.append(category_id)
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and "segmentation" in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        # add keypoints if requested; shape [N, num_keypoints, 3] as (x, y, visibility)
        # with x, y in absolute pixel coordinates (normalised later by ``Normalize``).
        if self.include_keypoints:
            k = self.num_keypoints
            per_instance = []
            for obj in anno:
                raw = obj.get("keypoints", []) or []
                kp = torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 3)
                fixed = torch.zeros((k, 3), dtype=torch.float32)
                n = min(kp.shape[0], k)
                if n > 0:
                    fixed[:n] = kp[:n]
                per_instance.append(fixed)
            if per_instance:
                keypoints = torch.stack(per_instance, dim=0)[keep]
            else:
                keypoints = torch.zeros((0, k, 3), dtype=torch.float32)
            target["keypoints"] = keypoints

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def _build_train_resize_config(
    scales: List[int],
    *,
    square: bool,
    max_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build the training resize pipeline as an Albumentations config list.

    Expresses the ``RandomSelect(resize_a, Compose([resize_b1, crop, resize_b2]))`` pattern as a config-driven
    ``OneOf``/``Sequential`` for use with :meth:`AlbumentationsWrapper.from_config`.

    Two branches are selected with equal probability:

    - **Option A** – direct resize to the target scale(s).
    - **Option B** – resize to an intermediate scale (400/500/600 px), crop,
      then resize to the target scale.

    Divisibility padding (rounding ``H``/``W`` up to a multiple of ``patch_size * num_windows``) is handled by the batch
    collator via :func:`~rfdetr.utilities.tensors.make_collate_fn`, not here.

    Args:
        scales: Target resize scales in pixels.
        square: If ``True``, produce square output using ``A.Resize``
            (one random scale from *scales*).  If ``False``, preserve aspect ratio using ``A.SmallestMaxSize`` with an
            optional long-side cap.
        max_size: Maximum long-side size for non-square resizes.  Defaults to
            ``1333`` when *square* is ``False``.

    Returns:
        A single-element list containing a ``OneOf`` config entry.
    """
    # Scales are (height, width) pairs; for a square model both components are equal.
    pairs = [as_pair(s) for s in scales]

    if square:
        option_a: Dict[str, Any] = {
            "OneOf": {
                "transforms": [{"Resize": {"height": h, "width": w}} for h, w in pairs],
            }
        }
        option_b: Dict[str, Any] = {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {
                        "OneOf": {
                            "transforms": [
                                {"RandomSizedCrop": {"min_max_height": [384, 600], "height": h, "width": w}}
                                for h, w in pairs
                            ],
                        }
                    },
                ]
            }
        }
    else:
        cap = max_size or 1333
        # This branch resizes by shortest side and caps the longest, so it only needs the short edge.
        short_sides = [min(h, w) for h, w in pairs]
        # SmallestMaxSize accepts a list and picks randomly — no OneOf needed
        size_param: Any = short_sides[0] if len(short_sides) == 1 else short_sides
        option_a = {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": size_param}},
                    {"LongestMaxSize": {"max_size": cap}},
                ]
            }
        }
        option_b = {
            "Sequential": {
                "transforms": [
                    {"SmallestMaxSize": {"max_size": [400, 500, 600]}},
                    {"RandomSizedCrop": {"min_max_height": [384, 600], "height": 384, "width": 384}},
                    {"SmallestMaxSize": {"max_size": size_param}},
                    {"LongestMaxSize": {"max_size": cap}},
                ]
            }
        }

    return [{"OneOf": {"transforms": [option_a, option_b]}}]


def _build_resize(config: List[Dict[str, Any]], image_set: str) -> List[Any]:
    """Build the resize stage, refusing to return an empty pipeline.

    :meth:`AlbumentationsWrapper.from_config` logs and skips transforms it cannot construct, which is tolerable for
    augmentations but not for the resize: dropping it leaves images at their native size while boxes are normalised
    against that size, so the model trains on a geometry inference never reproduces.
    """
    wrappers = AlbumentationsWrapper.from_config(config)
    if not wrappers:
        raise RuntimeError(
            f"Resize stage for image_set={image_set!r} built zero transforms from {config!r}. Training without it "
            "would feed the model images at their native size instead of the configured resolution."
        )
    return wrappers


def make_coco_transforms(
    image_set: str,
    resolution: int | Tuple[int, int],
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    aug_config: Optional[Dict[str, Dict[str, Any]]] = None,
    gpu_postprocess: bool = False,
) -> Compose:
    """Build the standard COCO transform pipeline for a given dataset split.

    Returns a composed transform that resizes images to the target ``resolution`` (with optional multi-scale jitter),
    applies Albumentations-based augmentations during training, and normalises pixel values with ImageNet statistics.

    For the ``"train"`` split the pipeline uses a two-branch ``OneOf`` between a direct resize and a resize →
    random-crop → resize sequence (built via :func:`_build_train_resize_config`), followed by the augmentation stack and
    normalisation.  For ``"val"``, ``"test"``, and ``"val_speed"`` only resize and normalisation are applied — no
    augmentation.

    When *gpu_postprocess* is ``True``, both the Albumentations augmentation wrappers and the ``Normalize`` step are
    omitted from the ``"train"`` pipeline. The ``RFDETRDataModule`` then applies augmentation and normalization on the
    device in ``on_after_batch_transfer`` instead.

    Args:
        image_set: Dataset split identifier — ``"train"``, ``"val"``, ``"test"``,
            or ``"val_speed"``.
        resolution: Target short-side resolution in pixels.  During validation the
            longest side is capped at 1333 px to preserve aspect ratio.
        multi_scale: If ``True``, sample the resize target from a range of scales
            computed by :func:`compute_multi_scale_scales` instead of using a single fixed size.
        expanded_scales: Passed to :func:`compute_multi_scale_scales`; broadens the
            scale range when ``multi_scale=True``.
        skip_random_resize: When ``multi_scale=True``, use only the largest scale
            and skip random selection among multiple scales.
        patch_size: Model patch size used by :func:`compute_multi_scale_scales` to
            ensure all candidate resolutions are compatible with the backbone.
        num_windows: Number of attention windows; used by
            :func:`compute_multi_scale_scales` to derive candidate resolutions.
        aug_config: Albumentations augmentation config dict passed to
            :class:`~rfdetr.datasets.transforms.AlbumentationsWrapper`.  Falls back to the default
            :data:`~rfdetr.datasets.aug_config.AUG_CONFIG` when ``None``.
        gpu_postprocess: When ``True``, skip Albumentations augmentation wrappers and
            ``Normalize`` from the CPU pipeline.  The ``RFDETRDataModule`` then applies both augmentation and
            normalization on the GPU in ``on_after_batch_transfer``.  Has no effect on val/test splits.

    Returns:
        A :class:`torchvision.transforms.v2.Compose` pipeline ready to be passed to :class:`CocoDetection`.

        .. note::
            This pipeline does **not** guarantee that output ``H`` and ``W`` are divisible by ``patch_size *
            num_windows``.  Divisibility is enforced at the batch level by the DataLoader collate function.  If you
            apply these transforms outside of :class:`~rfdetr.training.module_data.RFDETRDataModule`, pass the result
            through :func:`~rfdetr.utilities.tensors.nested_tensor_from_tensor_list` with ``block_size=patch_size *
            num_windows``, or use :func:`~rfdetr.utilities.tensors.make_collate_fn` with that value.

    Raises:
        ValueError: If ``image_set`` is not one of the recognised split names.
    """
    to_image = ToImage()
    to_float = ToDtype(torch.float32, scale=True)
    normalize = Normalize()

    scales = [as_pair(resolution)]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        logger.info(f"Using multi-scale training with scales: {scales}")

    if image_set == "train":
        resolved_aug_config = aug_config if aug_config is not None else AUG_CONFIG
        resize_wrappers = _build_resize(
            _build_train_resize_config(scales, square=False, max_size=1333), image_set
        )
        pipeline = [*resize_wrappers]
        if not gpu_postprocess:
            aug_wrappers = AlbumentationsWrapper.from_config(resolved_aug_config)
            pipeline += [*aug_wrappers]
        pipeline += [to_image, to_float]
        if not gpu_postprocess:
            pipeline += [normalize]
        return Compose(pipeline)

    if image_set in ("val", "test"):
        resize_wrappers = _build_resize(
            [
                {"SmallestMaxSize": {"max_size": min(as_pair(resolution))}},
                {"LongestMaxSize": {"max_size": 1333}},
            ],
            image_set,
        )
        return Compose([*resize_wrappers, to_image, to_float, normalize])
    if image_set == "val_speed":
        _res_h, _res_w = as_pair(resolution)
        resize_wrappers = _build_resize([{"Resize": {"height": _res_h, "width": _res_w}}], image_set)
        return Compose([*resize_wrappers, to_image, to_float, normalize])

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(
    image_set: str,
    resolution: int | Tuple[int, int],
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    aug_config: Optional[Dict[str, Dict[str, Any]]] = None,
    gpu_postprocess: bool = False,
) -> Compose:
    """Create COCO transforms with square resizing where the output size is divisible by 64.

    This function builds a torchvision-style transform pipeline for COCO images that resizes them to square shapes
    suitable for models that require spatial dimensions divisible by 64. It supports multi-scale training and optional
    random resizing and cropping for the training split.

    When *gpu_postprocess* is ``True``, both the Albumentations augmentation wrappers and the ``Normalize`` step are
    omitted from the ``"train"`` pipeline. The ``RFDETRDataModule`` then applies augmentation and normalization on the
    device in ``on_after_batch_transfer`` instead.

    Args:
        image_set: Dataset split identifier. Expected values are "train", "val",
            "test", or "val_speed". Each split uses a slightly different transform pipeline suited for training or
            evaluation.
        resolution: Base square resolution (in pixels) to which images are resized.
        multi_scale: If True, enable multi-scale training by sampling from a set of
            square resolutions instead of a single fixed size.
        expanded_scales: If True, expand the range of scales used during
            multi-scale training. Passed through to ``compute_multi_scale_scales``.
        skip_random_resize: If True and ``multi_scale`` is enabled, use only the
            largest scale returned by ``compute_multi_scale_scales`` and skip random selection among multiple scales.
        patch_size: Patch size used by ``compute_multi_scale_scales`` when
            determining valid square resolutions (typically related to the model's patch embedding or stride).
        num_windows: Number of windows used by ``compute_multi_scale_scales`` to
            derive the list of candidate square resolutions.
        aug_config: Augmentation configuration dictionary compatible with
            :class:`~rfdetr.datasets.transforms.AlbumentationsWrapper`. If ``None``, the default
            :data:`~rfdetr.datasets.aug_config.AUG_CONFIG` is used.
        gpu_postprocess: When ``True``, skip Albumentations augmentation wrappers and
            ``Normalize`` from the CPU pipeline.  The ``RFDETRDataModule`` then applies both augmentation and
            normalization on the GPU in ``on_after_batch_transfer``.  Has no effect on val/test splits.

    Returns:
        A ``Compose`` object containing the composed image transforms appropriate for the specified ``image_set``.
    """
    to_image = ToImage()
    to_float = ToDtype(torch.float32, scale=True)
    normalize = Normalize()

    scales = [as_pair(resolution)]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        logger.info(f"Using multi-scale training with square resize and scales: {scales}")

    if image_set == "train":
        resolved_aug_config = aug_config if aug_config is not None else AUG_CONFIG
        resize_wrappers = _build_resize(_build_train_resize_config(scales, square=True), image_set)
        pipeline = [*resize_wrappers]
        if not gpu_postprocess:
            aug_wrappers = AlbumentationsWrapper.from_config(resolved_aug_config)
            pipeline += [*aug_wrappers]
        pipeline += [to_image, to_float]
        if not gpu_postprocess:
            pipeline += [normalize]
        return Compose(pipeline)

    if image_set in ("val", "test", "val_speed"):
        _res_h, _res_w = as_pair(resolution)
        resize_wrappers = _build_resize([{"Resize": {"height": _res_h, "width": _res_w}}], image_set)
        return Compose([*resize_wrappers, to_image, to_float, normalize])

    raise ValueError(f"unknown {image_set}")


def build_coco(image_set: str, args: Any, resolution: int | Tuple[int, int]) -> CocoDetection:
    root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
    if not root.exists():
        logger.error(f"COCO path {root} does not exist")
        raise FileNotFoundError(f"COCO path {root} does not exist")

    mode = "instances"
    PATHS = {  # noqa: N806
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
        "test": (root / "test2017", root / "annotations" / "image_info_test-dev2017.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    include_keypoints = getattr(args, "keypoint_head", False)
    num_keypoints = getattr(args, "num_keypoints", 2)
    aug_config = getattr(args, "aug_config", None)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_augmentation_backend = _resolve_runtime_augmentation_backend(augmentation_backend)
    if resolved_augmentation_backend != augmentation_backend and resolved_augmentation_backend == "cpu":
        logger.warning(
            "augmentation_backend='auto' resolved to 'cpu' because CUDA or kornia is unavailable; "
            "disabling GPU postprocess transforms and retaining CPU normalization."
        )
    gpu_postprocess = resolved_augmentation_backend != "cpu"
    _require_cpu_backend_for_keypoints(include_keypoints, gpu_postprocess)

    if square_resize_div_64:
        logger.info(f"Building COCO {image_set} dataset with square resize at resolution {resolution}")
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints=num_keypoints,
        )
    else:
        logger.info(f"Building COCO {image_set} dataset at resolution {resolution}")
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints=num_keypoints,
        )
    return dataset


def _require_cpu_backend_for_keypoints(include_keypoints: bool, gpu_postprocess: bool) -> None:
    """Fail fast when keypoint training is combined with the GPU augmentation backend.

    The GPU (kornia) augmentation pipeline in ``RFDETRDataModule.on_after_batch_transfer`` transforms images, boxes,
    and masks but has no keypoint path, so keypoints would silently keep their pre-augmentation coordinates and train
    the model on a geometry inference never reproduces.  Keypoint training therefore requires the CPU (Albumentations)
    backend, which does transform keypoints.
    """
    if include_keypoints and gpu_postprocess:
        raise ValueError(
            "Keypoint training requires augmentation_backend='cpu'. The GPU/kornia augmentation pipeline does not "
            "transform keypoints, so boxes and keypoints would fall out of alignment. Set augmentation_backend='cpu'."
        )


def _resolve_runtime_augmentation_backend(backend: str) -> str:
    """Resolve ``augmentation_backend`` at runtime for dataset builders.

    Thin wrapper around :func:`rfdetr.datasets.kornia_transforms.resolve_augmentation_backend` kept for
    backward-compatibility with callers in ``yolo.py``.

    ``"auto"`` becomes ``"gpu"`` only when CUDA and Kornia are both available, otherwise ``"cpu"``. Explicit
    ``"cpu"``/``"gpu"`` values pass through.
    """
    from rfdetr.datasets.kornia_transforms import resolve_augmentation_backend

    return resolve_augmentation_backend(backend)


def build_roboflow_from_coco(image_set: str, args: Any, resolution: int | Tuple[int, int]) -> CocoDetection:
    """Build a Roboflow COCO-format dataset.

    This uses Roboflow's standard directory structure (train/valid/test folders with _annotations.coco.json).
    """
    root = Path(args.dataset_dir)
    if not root.exists():
        logger.error(f"Roboflow dataset path {root} does not exist")
        raise FileNotFoundError(f"Roboflow dataset path {root} does not exist")

    PATHS = {  # noqa: N806
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    include_keypoints = getattr(args, "keypoint_head", False)
    num_keypoints = getattr(args, "num_keypoints", 2)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    do_random_resize_via_padding = getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)
    resolved_augmentation_backend = _resolve_runtime_augmentation_backend(getattr(args, "augmentation_backend", "cpu"))
    gpu_postprocess = resolved_augmentation_backend != "cpu"
    _require_cpu_backend_for_keypoints(include_keypoints, gpu_postprocess)

    if square_resize_div_64:
        logger.info(f"Building Roboflow {image_set} dataset with square resize at resolution {resolution}")
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints=num_keypoints,
            remap_category_ids=True,
        )
    else:
        logger.info(f"Building Roboflow {image_set} dataset at resolution {resolution}")
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=multi_scale,
                expanded_scales=expanded_scales,
                skip_random_resize=not do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                aug_config=aug_config,
                gpu_postprocess=gpu_postprocess,
            ),
            include_masks=include_masks,
            include_keypoints=include_keypoints,
            num_keypoints=num_keypoints,
            remap_category_ids=True,
        )
    return dataset
