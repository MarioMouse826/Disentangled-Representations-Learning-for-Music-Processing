"""Device selection utilities.

All GPU-capable code paths in the project should default to `best_device()`
so a local Mac run picks up MPS while an HPC run picks up CUDA, without
per-callsite if/else.
"""
from __future__ import annotations

import os

import torch


def best_device(prefer: str | None = None) -> torch.device:
    """Pick the most capable available device.

    Order: explicit `prefer` → env var `SC_VAE_DEVICE` → CUDA → MPS → CPU.

    Args:
        prefer: if set to `"cuda"`, `"mps"`, or `"cpu"`, returns that device
            iff it is actually available; otherwise falls through.

    Returns:
        torch.device. Always a valid, initialized device.
    """
    requested = prefer or os.environ.get("SC_VAE_DEVICE")
    if requested:
        requested = requested.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        # Fall through — `prefer` was unavailable.

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Normalize a `str | torch.device | None | "auto"` arg into a `torch.device`.

    `None` and `"auto"` both route through `best_device()`.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return best_device()
    return torch.device(device)
