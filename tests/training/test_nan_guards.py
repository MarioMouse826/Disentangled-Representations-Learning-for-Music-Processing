from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.baselines.beta_vae import BetaVAE
from src.training.callbacks import NaNGuardCallback
from src.training.lit_module import VAELitModule


class _DS(Dataset):
    def __init__(self, n: int = 4, x_fill: float | None = None) -> None:
        torch.manual_seed(0)
        self._items = [
            torch.randn(1, 128, 401) if x_fill is None
            else torch.full((1, 128, 401), x_fill)
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return {"x": self._items[i]}


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# -- constructor ----------------------------------------------------------


def test_guard_default_action_is_raise() -> None:
    g = NaNGuardCallback()
    assert g.action == "raise"
    assert g.check_params is False  # expensive — off by default


def test_guard_invalid_action_raises() -> None:
    with pytest.raises(ValueError, match="action"):
        NaNGuardCallback(action="bogus")


# -- normal training ------------------------------------------------------


def test_guard_no_op_on_healthy_batch() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    guard = NaNGuardCallback(action="raise")
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        callbacks=[guard],
    )
    loader = DataLoader(_DS(n=2), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader, loader)  # must not raise
    assert guard.nan_count == 0


# -- NaN detection --------------------------------------------------------


def test_guard_raises_on_nan_loss() -> None:
    # Force NaN by corrupting model weights post-init → encoder output NaN
    # → loss NaN → guard raises.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    with torch.no_grad():
        model.to_mu.weight.fill_(float("nan"))
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    guard = NaNGuardCallback(action="raise")
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        callbacks=[guard],
    )
    loader = DataLoader(_DS(n=1), batch_size=1, collate_fn=_collate)

    with pytest.raises(RuntimeError, match="NaN"):
        trainer.fit(module, loader)


def test_guard_skip_mode_tolerates_nan() -> None:
    # action='skip' → log + zero grads + no raise. Training completes.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    with torch.no_grad():
        model.to_mu.weight.fill_(float("nan"))
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    guard = NaNGuardCallback(action="skip")
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        callbacks=[guard],
    )
    loader = DataLoader(_DS(n=1), batch_size=1, collate_fn=_collate)
    trainer.fit(module, loader)  # no raise
    assert guard.nan_count >= 1


# -- grad norm logging ----------------------------------------------------


def test_guard_tracks_grad_norm() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    guard = NaNGuardCallback()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        callbacks=[guard],
    )
    loader = DataLoader(_DS(n=4), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader)
    # grad_norm tracker populated with at least one non-None entry.
    assert guard.last_grad_norm is not None
    assert guard.last_grad_norm > 0
    assert torch.isfinite(torch.tensor(guard.last_grad_norm))


# -- anomaly-detection utility -------------------------------------------


def test_set_anomaly_detection_toggle() -> None:
    from src.training.callbacks import set_anomaly_detection

    with set_anomaly_detection(True):
        assert torch.is_anomaly_enabled()
    # Restored after context exit.
    assert torch.is_anomaly_enabled() is False


def test_set_anomaly_detection_nested() -> None:
    from src.training.callbacks import set_anomaly_detection

    with set_anomaly_detection(True):
        assert torch.is_anomaly_enabled()
        with set_anomaly_detection(False):
            assert torch.is_anomaly_enabled() is False
        assert torch.is_anomaly_enabled()
    assert torch.is_anomaly_enabled() is False


# -- param-check mode (expensive, opt-in) --------------------------------


def test_guard_checks_params_directly() -> None:
    # Unit-test the check_params scan: corrupt a param, call `_scan_parameters`,
    # verify it detects + raises. Avoids autograd in-place complications
    # that arise when corrupting params mid-Trainer-loop.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    module.log = lambda *a, **k: None  # type: ignore[assignment]
    guard = NaNGuardCallback(action="raise", check_params=True)

    with torch.no_grad():
        model.encoder.blocks[0].net[0].weight.data[0, 0, 0, 0] = float("inf")

    with pytest.raises(RuntimeError, match="NaN/Inf in parameter"):
        guard._scan_parameters(module, batch_idx=0)


def test_guard_scan_passes_on_clean_params() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    module.log = lambda *a, **k: None  # type: ignore[assignment]
    guard = NaNGuardCallback(action="raise", check_params=True)
    # Fresh init → no NaN/Inf anywhere; scan should not raise.
    guard._scan_parameters(module, batch_idx=0)
    assert guard.nan_count == 0


# -- skip-mode grad zeroing ----------------------------------------------


def test_guard_skip_mode_zeroes_bad_grad() -> None:
    # Unit-test that action="skip" actually zeroes bad gradients in-place on
    # p.grad (not on a detached view). Verifies the on_before_optimizer_step
    # path without needing a full Trainer loop.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    guard = NaNGuardCallback(action="skip")

    bad_param = None
    for p in module.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, float("inf"))
            bad_param = p
            break
    assert bad_param is not None

    # Patch log to no-op since no Trainer is attached.
    module.log = lambda *a, **k: None  # type: ignore[assignment]

    guard.on_before_optimizer_step(trainer=None, pl_module=module, optimizer=None)  # type: ignore[arg-type]

    assert torch.all(bad_param.grad == 0)
    assert guard.nan_count >= 1
