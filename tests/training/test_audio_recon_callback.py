from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset


class _FakeLogger(Logger):
    """Minimal pl.loggers.Logger with a MagicMock experiment — Lightning's
    `Trainer(logger=...)` validation rejects raw MagicMock, so subclass
    the abstract Logger and stub the required methods."""

    def __init__(self) -> None:
        super().__init__()
        self._experiment = MagicMock()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def version(self) -> str:
        return "0"

    @property
    def experiment(self) -> MagicMock:
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    @rank_zero_only
    def log_hyperparams(self, params: dict) -> None:
        pass

from src.models.baselines.beta_vae import BetaVAE
from src.models.sc_vae import SCVAE
from src.training.callbacks import AudioReconCallback
from src.training.lit_module import VAELitModule


class _SimpleDS(Dataset):
    def __init__(self, n: int = 4) -> None:
        torch.manual_seed(0)
        self._items = [torch.randn(1, 128, 401) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return {"x": self._items[i]}


def _collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# -- construct + config --------------------------------------------------


def test_callback_constructor_valid_args() -> None:
    cb = AudioReconCallback(num_samples=4)
    assert cb.num_samples == 4
    assert cb.log_waveform is False  # default: mel images only


def test_callback_rejects_invalid_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        AudioReconCallback(num_samples=0)
    with pytest.raises(ValueError, match="num_samples"):
        AudioReconCallback(num_samples=-1)


# -- logging path --------------------------------------------------------


def test_callback_logs_num_samples_artifacts() -> None:
    # Plan-critical: on validation-epoch-end, callback logs `num_samples`
    # reconstruction artifacts. We mock the WandbLogger's experiment
    # object + count .log() calls.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=3, log_waveform=False)

    fake_logger = _FakeLogger()

    # fast_dev_run suppresses logger calls — use minimal regular run.
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=fake_logger,
        accelerator="cpu",
        callbacks=[cb],
    )
    # Batch size 4 ≥ num_samples=3 so no clamping — expected output is
    # exactly 3 mel images.
    loader = DataLoader(_SimpleDS(n=4), batch_size=4, collate_fn=_collate)
    trainer.fit(module, loader, loader)

    # Every validation epoch end fires one consolidated log() call with a
    # dict that includes `num_samples` Image entries.
    assert fake_logger.experiment.log.called
    call_args_list = fake_logger.experiment.log.call_args_list
    last_payload = call_args_list[-1].args[0] if call_args_list[-1].args else call_args_list[-1].kwargs.get("data", {})
    # Keys like 'recon/mel_0', 'recon/mel_1', 'recon/mel_2'.
    image_keys = [k for k in last_payload.keys() if k.startswith("recon/mel_")]
    assert len(image_keys) == 3


def test_callback_skips_when_no_logger() -> None:
    # When run outside W&B (logger=False), callback must no-op instead of
    # crashing on a missing experiment handle.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=2)
    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        callbacks=[cb],
    )
    loader = DataLoader(_SimpleDS(n=2), batch_size=1, collate_fn=_collate)
    trainer.fit(module, loader, loader)  # must not raise


def test_callback_caps_num_samples_at_batch_size() -> None:
    # Asking for more samples than the validation batch provides is safe:
    # callback silently clamps.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=100)  # larger than batch

    fake_logger = _FakeLogger()

    # fast_dev_run suppresses logger calls — use minimal regular run.
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=fake_logger,
        accelerator="cpu",
        callbacks=[cb],
    )
    loader = DataLoader(_SimpleDS(n=2), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader, loader)

    payload = fake_logger.experiment.log.call_args.args[0]
    image_keys = [k for k in payload.keys() if k.startswith("recon/mel_")]
    assert len(image_keys) == 2  # clamped to batch size


def test_callback_works_on_sc_vae() -> None:
    # SC-VAE forward returns a superset of BetaVAE's output dict; verify
    # the callback pulls `x_hat` correctly from both.
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=2)

    fake_logger = _FakeLogger()

    # fast_dev_run suppresses logger calls — use minimal regular run.
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=fake_logger,
        accelerator="cpu",
        callbacks=[cb],
    )
    loader = DataLoader(_SimpleDS(n=2), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader, loader)

    assert fake_logger.experiment.log.called


def test_callback_freezes_sample_batch() -> None:
    # Plan spec: callback picks a *fixed* validation batch across epochs so
    # reconstructions are comparable across training. Verify by peeking
    # at the captured batch attribute after first validation.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=2)

    fake_logger = _FakeLogger()

    # fast_dev_run suppresses logger calls — use minimal regular run.
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=fake_logger,
        accelerator="cpu",
        callbacks=[cb],
    )
    loader = DataLoader(_SimpleDS(n=4), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader, loader)

    assert cb._fixed_batch is not None
    assert cb._fixed_batch["x"].shape == (2, 1, 128, 401)


def test_callback_log_waveform_flag_toggles_audio() -> None:
    # When log_waveform=True, payload should include 'recon/audio_*' keys
    # alongside the mel images.
    model = BetaVAE(d_z=8, n_mels=128, n_time=401)
    module = VAELitModule(model=model, total_steps=10, warmup_steps=1)
    cb = AudioReconCallback(num_samples=2, log_waveform=True, sample_rate=16000)

    fake_logger = _FakeLogger()

    # fast_dev_run suppresses logger calls — use minimal regular run.
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=fake_logger,
        accelerator="cpu",
        callbacks=[cb],
    )
    loader = DataLoader(_SimpleDS(n=2), batch_size=2, collate_fn=_collate)
    trainer.fit(module, loader, loader)

    payload = fake_logger.experiment.log.call_args.args[0]
    audio_keys = [k for k in payload.keys() if k.startswith("recon/audio_")]
    assert len(audio_keys) == 2
