"""Tests for deterministic seed setup."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_seeds


def test_set_seeds_python_random_reproducible() -> None:
    set_seeds(42)
    a = [random.random() for _ in range(5)]
    set_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seeds_numpy_reproducible() -> None:
    set_seeds(7)
    a = np.random.randn(10)
    set_seeds(7)
    b = np.random.randn(10)
    np.testing.assert_array_equal(a, b)


def test_set_seeds_torch_cpu_reproducible() -> None:
    set_seeds(99)
    a = torch.randn(8)
    set_seeds(99)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seeds_sets_cudnn_flags() -> None:
    set_seeds(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seeds_returns_seed_for_logging() -> None:
    returned = set_seeds(123)
    assert returned == 123


def test_set_seeds_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seeds(-1)
