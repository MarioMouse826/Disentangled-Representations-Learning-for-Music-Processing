"""Deterministic seed setup for reproducibility.

Covers `random`, `numpy`, `torch` (CPU + CUDA), cuDNN flags, and Python
hash-seed parity via the env var `PYTHONHASHSEED`. Call once at process
start before any RNG-dependent op.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int) -> int:
    """Set all relevant RNGs to `seed` and pin cuDNN to deterministic mode.

    Args:
        seed: non-negative integer.

    Returns:
        The seed that was set (handy for logging).
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN: deterministic kernels, no autotune. Costs throughput for
    # bitwise reproducibility — required by the paper's protocol.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
