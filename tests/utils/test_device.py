from __future__ import annotations

import os

import pytest
import torch

from src.utils.device import best_device, resolve_device


def test_best_device_returns_valid_device() -> None:
    dev = best_device()
    assert isinstance(dev, torch.device)
    # Constructing a tensor on the returned device must succeed.
    torch.zeros(1, device=dev)


def test_prefer_cpu_always_respected() -> None:
    dev = best_device(prefer="cpu")
    assert dev.type == "cpu"


def test_prefer_invalid_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SC_VAE_DEVICE", raising=False)
    # 'cuda' fallback when CUDA unavailable must still return a valid device.
    dev = best_device(prefer="cuda")
    assert isinstance(dev, torch.device)


def test_env_var_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SC_VAE_DEVICE", "cpu")
    assert best_device().type == "cpu"


def test_resolve_device_auto() -> None:
    assert resolve_device("auto") == best_device()
    assert resolve_device(None) == best_device()


def test_resolve_device_string() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_passthrough() -> None:
    cpu = torch.device("cpu")
    assert resolve_device(cpu) == cpu


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable on this box",
)
def test_mps_path_available() -> None:
    # Smoke: the "best device" on an Apple-Silicon dev box must be MPS,
    # not CPU, when CUDA is unavailable.
    if not torch.cuda.is_available():
        assert best_device().type == "mps"
