from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.baselines.beta_vae import BetaVAE
from src.models.sc_vae import SCVAE
from src.training.lit_module import VAELitModule


# -- tiny dummy datasets -------------------------------------------------


class _SimpleDataset(Dataset):
    """Yields `{"x": (1, 128, 401)}` dicts — mimics collate output for baselines."""

    def __init__(self, n: int = 8) -> None:
        torch.manual_seed(0)
        self._items = [torch.randn(1, 128, 401) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return {"x": self._items[i]}


class _PairedDataset(Dataset):
    """Yields paired `(x, x_g, g_cents)` batches — mimics PairedPitchShiftCollate."""

    def __init__(self, n: int = 8) -> None:
        torch.manual_seed(0)
        self._items = [
            {
                "x": torch.randn(1, 128, 401),
                "x_g": torch.randn(1, 128, 401),
                "g_cents": torch.tensor(100 * (i % 5 - 2), dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return self._items[i]


def _collate(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# -- core smoke tests ----------------------------------------------------


def test_fast_dev_run_on_beta_vae() -> None:
    # Plan-critical: Trainer(fast_dev_run=True).fit runs cleanly on a tiny
    # batch. Exercises training_step + configure_optimizers wiring.
    model = BetaVAE(d_z=16, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=100, warmup_steps=10)
    loader = DataLoader(_SimpleDataset(n=4), batch_size=2, collate_fn=_collate)
    trainer = pl.Trainer(
        fast_dev_run=True, enable_progress_bar=False, logger=False, accelerator="cpu"
    )
    trainer.fit(module, loader, loader)


def test_fast_dev_run_on_sc_vae_paired() -> None:
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model,
        total_steps=100,
        warmup_steps=10,
        lambda_inv=1.0,
        lambda_equi=1.0,
        lambda_swap=0.5,
    )
    loader = DataLoader(_PairedDataset(n=4), batch_size=2, collate_fn=_collate)
    trainer = pl.Trainer(
        fast_dev_run=True, enable_progress_bar=False, logger=False, accelerator="cpu"
    )
    trainer.fit(module, loader, loader)


def test_fast_dev_run_on_sc_vae_no_pairing() -> None:
    # SC-VAE with symmetry-loss lambdas at 0 + simple batch should still
    # run as pure recon + KL model (collapses to β-VAE-style step).
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model,
        total_steps=100,
        warmup_steps=10,
        lambda_inv=0.0,
        lambda_equi=0.0,
        lambda_swap=0.0,
    )
    loader = DataLoader(_SimpleDataset(n=4), batch_size=2, collate_fn=_collate)
    trainer = pl.Trainer(
        fast_dev_run=True, enable_progress_bar=False, logger=False, accelerator="cpu"
    )
    trainer.fit(module, loader, loader)


# -- optimizer + scheduler ----------------------------------------------


def test_configure_optimizers_returns_adamw_plus_cosine() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model, lr=1e-4, weight_decay=1e-6, total_steps=100, warmup_steps=10
    )
    opts = module.configure_optimizers()
    assert isinstance(opts, dict)
    assert "optimizer" in opts and "lr_scheduler" in opts
    opt = opts["optimizer"]
    assert isinstance(opt, torch.optim.AdamW)
    # Pin base hyperparameters from the plan spec. `opt.defaults` is
    # immune to scheduler mutations of `param_groups[0]["lr"]`.
    assert opt.defaults["lr"] == 1e-4
    group = opt.param_groups[0]
    assert group["weight_decay"] == 1e-6
    assert tuple(group["betas"]) == (0.9, 0.99)


def test_lr_schedule_warmup_then_cosine() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model, lr=1.0, total_steps=100, warmup_steps=10
    )
    sched = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    # Step 0 → near-zero LR (warmup start).
    lrs = []
    for _ in range(11):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    assert lrs[0] < 0.15                              # warmup start
    assert 0.9 < lrs[10] <= 1.01                      # warmup end ≈ base lr
    # After warmup, cosine decay → LR drops.
    for _ in range(80):
        sched.step()
    assert sched.get_last_lr()[0] < lrs[10]


# -- loss logging + composite parts -------------------------------------


def test_training_step_logs_all_loss_components() -> None:
    # Capture Lightning's log() calls via monkeypatch.
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model,
        total_steps=100,
        warmup_steps=10,
        lambda_inv=1.0, lambda_equi=1.0, lambda_swap=0.5,
    )
    logged: dict[str, float] = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value) if torch.is_tensor(value) else value

    module.log = fake_log  # type: ignore[assignment]

    batch = _collate([_PairedDataset(n=2)[i] for i in range(2)])
    loss = module.training_step(batch, 0)
    assert torch.is_tensor(loss)
    # Every component must be logged.
    for key in ("train/loss", "train/recon", "train/kl_s", "train/kl_c"):
        assert key in logged, f"missing log: {key}"
    # Symmetry logs appear when lambdas > 0.
    for key in ("train/L_inv", "train/L_equi", "train/L_swap"):
        assert key in logged, f"missing log: {key}"


def test_validation_step_logs_val_prefix() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=100, warmup_steps=10)
    logged: dict[str, float] = {}
    module.log = lambda name, value, **kw: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )  # type: ignore[assignment]

    batch = _collate([_SimpleDataset(n=2)[i] for i in range(2)])
    module.validation_step(batch, 0)
    assert "val/loss" in logged


# -- config ---------------------------------------------------------------


def test_config_includes_hyperparameters() -> None:
    model = BetaVAE(d_z=16, n_mels=128, n_time=401)
    module = VAELitModule(
        model=model,
        lr=2e-4,
        weight_decay=1e-5,
        beta_s=4.0,
        beta_c=2.0,
        tau_s=0.2,
        tau_c=0.05,
        lambda_inv=1.0,
        lambda_equi=1.0,
        lambda_swap=0.5,
        warmup_steps=2000,
        total_steps=100000,
    )
    cfg = module.config
    assert cfg["lr"] == 2e-4
    assert cfg["beta_s"] == 4.0
    assert cfg["beta_c"] == 2.0
    assert cfg["lambda_swap"] == 0.5
    assert cfg["total_steps"] == 100000


# -- invalid args --------------------------------------------------------


def test_invalid_lr_raises() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="lr"):
        VAELitModule(model=model, lr=-0.1, total_steps=100)


def test_invalid_total_steps_raises() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="total_steps"):
        VAELitModule(model=model, total_steps=0)


def test_invalid_warmup_steps_raises() -> None:
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="warmup_steps"):
        VAELitModule(model=model, warmup_steps=200, total_steps=100)
