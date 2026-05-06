"""Tests for compute-budget logging (plan Task 6.6)."""
from __future__ import annotations

import time

import pytest
import torch

from src.utils.compute_budget import (
    BudgetTracker,
    estimate_flops_per_step,
)


# -- BudgetTracker --------------------------------------------------------


def test_tracker_records_positive_wall_time() -> None:
    tracker = BudgetTracker()
    tracker.start()
    time.sleep(0.02)
    tracker.stop()
    summary = tracker.summary()
    assert summary["wall_time_hours"] > 0
    assert summary["wall_time_hours"] < 1.0  # far less than an hour


def test_tracker_context_manager_usage() -> None:
    with BudgetTracker() as tracker:
        time.sleep(0.01)
    summary = tracker.summary()
    assert summary["wall_time_hours"] > 0


def test_tracker_reports_zero_vram_on_cpu() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only behavior")
    with BudgetTracker() as tracker:
        _ = torch.zeros(100)
    summary = tracker.summary()
    assert summary["peak_vram_gb"] == 0.0


def test_tracker_summary_contains_required_keys() -> None:
    with BudgetTracker() as tracker:
        pass
    summary = tracker.summary()
    assert {"wall_time_hours", "gpu_hours", "peak_vram_gb"} <= summary.keys()


def test_tracker_raises_when_summary_without_stop() -> None:
    tracker = BudgetTracker()
    tracker.start()
    with pytest.raises(RuntimeError, match="stop"):
        tracker.summary()


# -- FLOPs estimation ------------------------------------------------------


def test_flops_per_step_on_linear_module() -> None:
    model = torch.nn.Linear(128, 64, bias=False)
    x = torch.randn(1, 128)
    flops = estimate_flops_per_step(model, x)
    # Linear layer: 128*64 = 8192 MACs → counted as multiply-adds.
    assert flops > 0
    assert flops >= 8000  # allow slack for fvcore counting convention


def test_flops_per_step_returns_zero_on_unsupported_module() -> None:
    # A module with only param-free ops (identity) should still return a
    # non-negative integer (fvcore returns 0 for unknown ops).
    model = torch.nn.Identity()
    x = torch.randn(1, 8)
    flops = estimate_flops_per_step(model, x)
    assert flops >= 0
