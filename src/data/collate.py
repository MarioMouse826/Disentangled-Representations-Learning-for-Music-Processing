"""Paired collate + DataLoader builder for SC-VAE training.

Every SC-VAE training step consumes a triplet `(x, T_g x, g)` where `T_g` is
the pitch-shift group action (Task 1.5). `PairedPitchShiftCollate` generates
`T_g x` per batch element inside the collate, so it runs inside DataLoader
worker processes and its CPU cost is hidden behind prefetch.

Usage::

    from src.data.augment import PitchShiftGroup
    from src.data.collate import build_paired_loader

    aug = PitchShiftGroup(sample_rate=16000)
    loader = build_paired_loader(
        dataset, aug=aug, batch_size=32, num_workers=8, seed=0,
    )
    for batch in loader:
        # batch["x"]        : (B, 1, T)   — original waveform
        # batch["x_g"]      : (B, 1, T)   — T_g(x)
        # batch["g_cents"]  : (B,)        — cents shift per item
        # batch["labels"]   : dict[str, Tensor] stacked from base dataset

GPU-first note
--------------
For the hot training path on HPC, offline-cache the (x, T_g x, g) triplets
instead of calling rubberband per step — rubberband is CPU-only and
subprocess-per-call, so a 32-GPU job is bottlenecked on 32 CPU threads
fighting for rubberband invocations. The live collate is fine for small-
scale runs; switch to a pre-generated shard loader for production training.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augment import PitchShiftGroup


# Item yielded by the base dataset. First element is waveform tensor
# `(C, T)`; second is a dict of per-item labels (each value is a scalar
# tensor).
DatasetItem = tuple[torch.Tensor, dict[str, torch.Tensor]]


class PairedPitchShiftCollate:
    """Collate callable producing paired `(x, T_g x, g)` batches.

    Args:
        aug: instance of `PitchShiftGroup`.
        seed: base RNG seed. Combined with the batch's minimum item index to
            derive a per-batch seed — keeps output reproducible across DataLoader
            workers while still varying batch-to-batch.
        range_cents: inclusive cents range for sampling `g`.
        quantum: `g` must be divisible by this value (50 = quarter-tone).
        per_item_g: if True, sample a different `g` per batch element; if
            False, apply the same `g` to every item in the batch (this is what
            the plan's `L_inv` requires — all batch items share transformation).
    """

    def __init__(
        self,
        aug: PitchShiftGroup,
        seed: int = 0,
        range_cents: tuple[int, int] = (-1200, 1200),
        quantum: int = 50,
        per_item_g: bool = False,
    ) -> None:
        self.aug = aug
        self.seed = int(seed)
        self.range_cents = range_cents
        self.quantum = quantum
        self.per_item_g = per_item_g
        # Zero-range special case: lo == hi short-circuits to a constant
        # draw of `lo`, avoiding a failure in `sample_g` (requires lo < hi).
        self._zero_range = range_cents[0] == range_cents[1]
        # Monotonic batch counter — content-independent so RNG does not
        # drift when `DataLoader(shuffle=True)` reshuffles batch composition.
        # In multi-worker mode each worker process gets its own counter
        # (fork/spawn gives each worker a private copy); we mix in
        # `torch.utils.data.get_worker_info().id` inside `_derive_rng` so
        # across workers the draws stay distinct but deterministic given
        # `(seed, worker_id, batch_idx_within_worker)`.
        self._batch_counter: int = 0

    def __call__(self, items: list[DatasetItem]) -> dict[str, Any]:
        if not items:
            raise ValueError("empty batch passed to PairedPitchShiftCollate")
        waves = [it[0] for it in items]
        labels = [it[1] for it in items]

        # Stack waveforms to (B, C, T). All items in a batch must share
        # shape already — upstream Dataset guarantees fixed-length output.
        x = torch.stack(waves, dim=0)
        B = x.shape[0]

        rng = self._derive_rng(items)
        g_cents = self._sample_g_batch(rng, B)

        # Apply T_g per item. Done on CPU; aug.apply is device-preserving
        # so output lands on the same device as the input (CPU at this point
        # since DataLoader workers collate on CPU).
        x_g = torch.empty_like(x)
        for i in range(B):
            x_g[i] = self.aug.apply(x[i], int(g_cents[i]))

        return {
            "x": x,
            "x_g": x_g,
            "g_cents": g_cents,
            "labels": self._stack_labels(labels),
        }

    # -- helpers ---------------------------------------------------------

    def _derive_rng(self, items: list[DatasetItem]) -> np.random.Generator:
        """RNG seeded with `(seed, worker_id, batch_idx)` — content-independent.

        Content-hash seeding (our earlier approach) drifts when
        `DataLoader(shuffle=True)` changes batch composition between runs,
        which silently breaks the 5-seed reproducibility protocol. Instead
        we derive from a monotonic per-worker batch counter. The counter is
        advanced *before* use so the first batch in each worker has index 0.
        """
        del items  # kept for API symmetry; not used.
        worker = torch.utils.data.get_worker_info()
        # numpy's SeedSequence rejects negative ints; use 2**32-1 as the
        # "no-worker" sentinel so main-process DataLoader runs still differ
        # from worker-0 (which is 0).
        worker_id = worker.id if worker is not None else (2**32 - 1)
        idx = self._batch_counter
        self._batch_counter += 1
        return np.random.default_rng((self.seed, worker_id, idx))

    def _sample_g_batch(self, rng: np.random.Generator, B: int) -> torch.Tensor:
        if self._zero_range:
            val = self.range_cents[0]
            return torch.full((B,), val, dtype=torch.long)
        if self.per_item_g:
            gs = [
                self.aug.sample_g(
                    rng=rng, range_cents=self.range_cents, quantum=self.quantum
                )
                for _ in range(B)
            ]
        else:
            g = self.aug.sample_g(
                rng=rng, range_cents=self.range_cents, quantum=self.quantum
            )
            gs = [g] * B
        return torch.tensor(gs, dtype=torch.long)

    @staticmethod
    def _stack_labels(
        labels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not labels:
            return {}
        # Preserve first-item insertion order — `set` drops ordering, which
        # leaks non-determinism into any downstream code that iterates
        # `batch["labels"].values()`.
        keys = list(labels[0].keys())
        reference = set(keys)
        for lbl in labels[1:]:
            if set(lbl.keys()) != reference:
                raise ValueError(
                    f"inconsistent label keys across batch: {reference} vs {set(lbl.keys())}"
                )
        stacked: dict[str, Any] = {}
        for k in keys:
            vals = [lbl[k] for lbl in labels]
            if torch.is_tensor(vals[0]):
                # Guard against ragged 1-D labels (e.g. ZeroShotLabel["beat_grid"])
                # which cannot be collated by torch.stack. Upstream should either
                # drop the field or use pad_sequence. Surface the contract
                # violation explicitly instead of crashing inside torch.stack
                # with a cryptic message.
                shape0 = vals[0].shape
                if any(v.shape != shape0 for v in vals[1:]):
                    raise ValueError(
                        f"label {k!r} has ragged shapes across batch "
                        f"(first={shape0}); use pad_sequence or drop the field"
                    )
                stacked[k] = torch.stack(vals, dim=0)
            elif isinstance(vals[0], str):
                # String labels (e.g. MoisesDBLabel.track_id, stem) can't be
                # tensor-collated; keep as a list. Downstream consumers treat
                # these as per-sample metadata, not tensor inputs.
                stacked[k] = vals  # type: ignore[assignment]
            else:
                stacked[k] = torch.as_tensor(vals)
        return stacked


def build_paired_loader(
    dataset: Dataset,
    *,
    aug: PitchShiftGroup,
    batch_size: int = 32,
    num_workers: int = 8,
    seed: int = 0,
    shuffle: bool = True,
    range_cents: tuple[int, int] = (-1200, 1200),
    quantum: int = 50,
    per_item_g: bool = False,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    collate_override: Callable[[list[DatasetItem]], dict[str, Any]] | None = None,
) -> DataLoader:
    """Factory wrapping DataLoader with PairedPitchShiftCollate defaults.

    Defaults align with the plan's recipe (`num_workers=8, persistent_workers=True`).
    `pin_memory` is auto-enabled when CUDA is available — MPS does not
    benefit from pinned memory so we default it off there.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    collate = collate_override or PairedPitchShiftCollate(
        aug=aug,
        seed=seed,
        range_cents=range_cents,
        quantum=quantum,
        per_item_g=per_item_g,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        # drop_last=True: β-TCVAE's `batch_tc_decomposition` requires B>=2; a
        # B=1 tail batch would crash mid-epoch. Drop tail for all models for
        # uniform training-step count; eval loaders can override.
        drop_last=True,
    )
