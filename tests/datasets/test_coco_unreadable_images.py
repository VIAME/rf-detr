# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for tolerating unreadable images in ``CocoDetection``.

An ``OSError`` escaping ``__getitem__`` runs inside a DataLoader worker, which
takes its DDP rank down; every surviving rank then blocks in the next collective
until the NCCL watchdog aborts the whole job one process-group timeout later.
These tests pin the behaviour that keeps a single bad read from doing that.
"""

from typing import Any, List, Tuple

import pytest

from rfdetr.datasets import coco as coco_module
from rfdetr.datasets.coco import CocoDetection


class _StubDataset(CocoDetection):
    """A ``CocoDetection`` whose ``_fetch`` fails for a chosen set of indices.

    Bypasses ``__init__`` so no annotation file or imagery is needed; only the
    attributes ``__getitem__`` actually touches are set.
    """

    def __init__(self, size: int, failing: set, exc: Exception = None):
        self.ids = list(range(size))
        self._failing = failing
        self._exc = exc or OSError("image file is truncated")
        self.fetched: List[int] = []

    def _fetch(self, idx: int) -> Tuple[Any, Any]:
        self.fetched.append(idx)
        if idx in self._failing:
            raise self._exc
        return (f"image-{idx}", {"image_id": idx})

    def _image_path(self, idx: int) -> str:
        return f"/data/frame{idx:06d}.png"


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep retry tests instant."""
    monkeypatch.setattr(coco_module.time, "sleep", lambda _: None)


class TestReadableImage:
    """A healthy sample is returned directly."""

    def test_returns_the_requested_sample(self):
        dataset = _StubDataset(size=10, failing=set())
        image, target = dataset[3]
        assert (image, target["image_id"]) == ("image-3", 3)

    def test_does_not_retry(self):
        dataset = _StubDataset(size=10, failing=set())
        dataset[3]
        assert dataset.fetched == [3]


class _FlakyDataset(_StubDataset):
    """Fails the first ``fail_times`` reads of an index, then succeeds.

    Mimics a short read off network storage, which raises the same ``OSError``
    PIL raises for a genuinely truncated file.
    """

    def __init__(self, size: int, flaky_idx: int, fail_times: int):
        super().__init__(size=size, failing=set())
        self._flaky_idx = flaky_idx
        self._remaining = fail_times

    def _fetch(self, idx: int) -> Tuple[Any, Any]:
        self.fetched.append(idx)
        if idx == self._flaky_idx and self._remaining > 0:
            self._remaining -= 1
            raise OSError("image file is truncated")
        return (f"image-{idx}", {"image_id": idx})


class TestTransientFailure:
    """A read that succeeds on retry yields the originally requested sample."""

    def test_returns_the_real_sample_after_one_failure(self):
        dataset = _FlakyDataset(size=10, flaky_idx=3, fail_times=1)
        image, target = dataset[3]
        assert (image, target["image_id"]) == ("image-3", 3)

    def test_retries_the_same_index_not_a_neighbour(self):
        dataset = _FlakyDataset(size=10, flaky_idx=3, fail_times=1)
        dataset[3]
        assert dataset.fetched == [3, 3]

    def test_survives_failures_up_to_the_attempt_limit(self):
        dataset = _FlakyDataset(
            size=10, flaky_idx=3, fail_times=coco_module._IMAGE_READ_ATTEMPTS - 1
        )
        image, _ = dataset[3]
        assert image == "image-3"


class TestPersistentFailure:
    """A permanently unreadable image is substituted, never raised."""

    def test_substitutes_the_next_index(self):
        dataset = _StubDataset(size=10, failing={3})
        image, target = dataset[3]
        assert (image, target["image_id"]) == ("image-4", 4)

    def test_retries_before_substituting(self):
        dataset = _StubDataset(size=10, failing={3})
        dataset[3]
        assert dataset.fetched.count(3) == coco_module._IMAGE_READ_ATTEMPTS

    def test_skips_consecutive_bad_images(self):
        dataset = _StubDataset(size=10, failing={3, 4, 5})
        image, _ = dataset[3]
        assert image == "image-6"

    def test_wraps_around_the_end_of_the_dataset(self):
        dataset = _StubDataset(size=4, failing={3})
        image, _ = dataset[3]
        assert image == "image-0"

    def test_returns_one_sample_per_call(self):
        """Batch counts must stay identical across ranks, so length is unchanged."""
        dataset = _StubDataset(size=10, failing={3})
        assert len(dataset[3]) == 2


class TestWholesaleFailure:
    """A broken dataset still surfaces as an error rather than silent duplication."""

    def test_raises_when_every_candidate_fails(self):
        dataset = _StubDataset(size=10, failing=set(range(10)))
        with pytest.raises(RuntimeError, match="consecutive samples were unreadable"):
            dataset[0]

    def test_error_names_the_first_bad_file(self):
        dataset = _StubDataset(size=10, failing=set(range(10)))
        with pytest.raises(RuntimeError, match=r"/data/frame000000\.png"):
            dataset[0]


class TestNonImageErrors:
    """Annotation and transform bugs must not be masked by substitution."""

    def test_value_error_propagates(self):
        dataset = _StubDataset(size=10, failing={3}, exc=ValueError("bad annotation"))
        with pytest.raises(ValueError, match="bad annotation"):
            dataset[3]

    def test_value_error_is_not_retried(self):
        dataset = _StubDataset(size=10, failing={3}, exc=ValueError("bad annotation"))
        with pytest.raises(ValueError):
            dataset[3]
        assert dataset.fetched == [3]
