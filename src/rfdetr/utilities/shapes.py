# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Helpers for handling square and non-square spatial shapes.

Input resolutions and patch grids are expressed either as a single ``int`` (square, the historical form) or as an
explicit ``(height, width)`` pair.  These helpers normalize the two forms so callers can work in ``(h, w)`` terms
without re-deriving a grid from a token count -- deriving a grid via ``sqrt`` silently produces the wrong answer for
non-square inputs.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["as_pair", "is_square"]


def as_pair(value: int | Iterable[int]) -> tuple[int, int]:
    """Normalize a size to an explicit ``(height, width)`` pair.

    Args:
        value: Either a single ``int`` (interpreted as a square ``(v, v)``) or an iterable
            of exactly two ints ordered ``(height, width)``.

    Returns:
        The size as a ``(height, width)`` tuple of ints.

    Raises:
        ValueError: If *value* is an iterable whose length is not exactly 2, or if any
            component is not a positive integer.
    """
    if isinstance(value, int):
        pair = (value, value)
    else:
        items = tuple(value)
        if len(items) != 2:
            raise ValueError(f"Expected an int or an iterable of 2 ints (height, width), got {value!r}.")
        pair = (int(items[0]), int(items[1]))

    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"Sizes must be positive, got {pair!r}.")
    return pair


def is_square(value: int | Iterable[int]) -> bool:
    """Return whether *value* denotes a square shape.

    Args:
        value: An ``int`` or a ``(height, width)`` pair.

    Returns:
        ``True`` when height equals width.
    """
    height, width = as_pair(value)
    return height == width
