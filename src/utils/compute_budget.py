"""Compute-budget logging (plan Task 6.6).

    wall_time_hours  - wall-clock elapsed since tracker start.
    gpu_hours        - wall_time_hours * num_visible_cuda_devices.
    peak_vram_gb     - peak allocated VRAM across all devices, reset at start.
    flops_per_step   - via fvcore.nn.FlopCountAnalysis.

Every run should wrap its main loop in `BudgetTracker` and write the
summary into `run_manifest.json` alongside the fields emitted by
`src.utils.manifest.build_manifest`.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import torch


_BYTES_PER_GB = 1024 ** 3


class BudgetTracker:
    """Records wall time + peak VRAM over a training / inference run.

    Usage:
        tracker = BudgetTracker()
        tracker.start()
        # ... training loop ...
        tracker.stop()
        summary = tracker.summary()    # dict with budget fields

    Or as a context manager:
        with BudgetTracker() as tracker:
            # ... run ...
        summary = tracker.summary()
    """

    def __init__(self) -> None:
        self._t_start: float | None = None
        self._t_stop: float | None = None

    def start(self) -> "BudgetTracker":
        self._t_start = time.perf_counter()
        self._t_stop = None
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                except RuntimeError:
                    # Device may not be visible/accessible; skip
                    pass
        return self

    def stop(self) -> "BudgetTracker":
        if self._t_start is None:
            raise RuntimeError("BudgetTracker.start() was not called")
        self._t_stop = time.perf_counter()
        return self

    def __enter__(self) -> "BudgetTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        if self._t_start is None or self._t_stop is None:
            raise RuntimeError(
                "BudgetTracker: call stop() before summary(); "
                "did you forget to close the context?"
            )
        wall_s = self._t_stop - self._t_start
        wall_h = wall_s / 3600.0

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            peak_bytes = 0
            for i in range(num_devices):
                try:
                    peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(i))
                except RuntimeError:
                    # Device may not be visible/accessible; skip
                    pass
            peak_gb = peak_bytes / _BYTES_PER_GB
            gpu_hours = wall_h * num_devices
        else:
            num_devices = 0
            peak_gb = 0.0
            gpu_hours = 0.0

        return {
            "wall_time_hours": float(wall_h),
            "gpu_hours": float(gpu_hours),
            "peak_vram_gb": float(peak_gb),
            "num_cuda_devices": int(num_devices),
        }


def estimate_flops_per_step(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
) -> int:
    """Count FLOPs for one forward pass via fvcore.

    Returns total MAC/FLOP count (fvcore counts multiply-adds; caller can
    double for strict FLOPs if needed). Returns 0 when fvcore declines to
    count (unknown op set).
    """
    from fvcore.nn import FlopCountAnalysis

    inputs = example_input if isinstance(example_input, tuple) else (example_input,)
    was_training = model.training
    model.train(False)
    try:
        with warnings.catch_warnings():
            # fvcore emits UserWarning for unsupported ops; surface total
            # only, suppress noise during counting.
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, inputs)
            fca = fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            total = int(fca.total())
    finally:
        if was_training:
            model.train()
    return max(total, 0)
