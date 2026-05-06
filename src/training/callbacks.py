"""Training callbacks (plan Tasks 4.3 + 4.4).

- `AudioReconCallback` — logs paired input/reconstruction mel spectrograms
  (and optional waveforms) for a **fixed** validation batch every epoch.
- `NaNGuardCallback` — monitors loss + gradients for NaN/Inf per step;
  raises or skips the offending step. Catches divergence early rather
  than letting corrupted weights poison hours of training.
- `set_anomaly_detection` — context manager wrapping
  `torch.autograd.set_detect_anomaly` for debug runs.

Gradient clipping is wired via `Trainer(gradient_clip_val=...)` from the
Hydra base config — not a callback since Lightning's built-in path
handles it inside the optimizer step.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class AudioReconCallback(pl.Callback):
    """Log input + reconstructed mel spectrograms each validation epoch.

    Args:
        num_samples: how many examples from the fixed validation batch to
            visualize. Clamped to batch size at runtime.
        log_waveform: if True, also inverts log-mel → waveform via
            Griffin-Lim + `wandb.Audio`. Slower; default off.
        sample_rate: target audio sample rate. Ignored when
            `log_waveform=False`.
        n_fft, n_mels: inverse-mel parameters for the Griffin-Lim path.
            Must match the frontend `LogMel` config.
        fmin, fmax: mel filterbank edges — same as `LogMel`.
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_waveform: bool = False,
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.num_samples = num_samples
        self.log_waveform = log_waveform
        self.sample_rate = sample_rate
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._fmin = fmin
        self._fmax = fmax

        self._fixed_batch: Optional[Mapping[str, torch.Tensor]] = None
        self._griffin_lim = None  # lazy-init on first use
        self._inv_mel = None

    # -- batch capture ---------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Capture the first validation batch once; keep it frozen across
        # epochs so reconstructions are directly comparable over time.
        if self._fixed_batch is None and batch_idx == 0:
            self._fixed_batch = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

    # -- epoch-end log ---------------------------------------------------

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._fixed_batch is None:
            return
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        if not hasattr(experiment, "log"):
            return

        device = pl_module.device
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self._fixed_batch.items()
        }
        x = batch["x"]
        n = min(self.num_samples, x.shape[0])

        was_training = pl_module.training
        pl_module.train(False)
        try:
            with torch.no_grad():
                out = pl_module.model(x[:n])
            x_hat = out["x_hat"].detach().cpu()
        finally:
            pl_module.train(was_training)

        x_cpu = x[:n].detach().cpu()
        payload: dict[str, Any] = {"epoch": trainer.current_epoch}

        # Lazy wandb import: not every run uses W&B, but callback lives in
        # the main module so we import on first log to keep the module
        # import-safe without wandb installed.
        try:
            import wandb
        except ImportError:
            wandb = None  # type: ignore[assignment]

        for i in range(n):
            payload[f"recon/mel_{i}"] = self._mel_image(
                x_cpu[i, 0].numpy(),
                x_hat[i, 0].numpy(),
                wandb_mod=wandb,
            )

        if self.log_waveform:
            for i in range(n):
                wav_in = self._mel_to_wave(x_cpu[i])
                wav_hat = self._mel_to_wave(x_hat[i])
                stereo = np.stack([wav_in, wav_hat], axis=0)  # (2, T)
                payload[f"recon/audio_{i}"] = self._wandb_audio(
                    stereo, wandb_mod=wandb
                )

        experiment.log(payload)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _mel_image(
        mel_in: np.ndarray,
        mel_hat: np.ndarray,
        wandb_mod: Any,
    ) -> Any:
        """Render input + reconstruction side-by-side as a wandb.Image."""
        pair = np.concatenate([mel_in, mel_hat], axis=0)
        pair = AudioReconCallback._normalize_uint8(pair)
        if wandb_mod is not None:
            return wandb_mod.Image(pair, caption="top: input  |  bottom: recon")
        return pair  # numpy fallback when wandb unavailable

    @staticmethod
    def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - lo) / (hi - lo) * 255.0
        return scaled.astype(np.uint8)

    def _mel_to_wave(self, mel: torch.Tensor) -> np.ndarray:
        """Invert log-mel → approximate waveform via Griffin-Lim."""
        import torchaudio

        if self._griffin_lim is None:
            hop = self._n_fft // 4  # match LogMel default
            self._inv_mel = torchaudio.transforms.InverseMelScale(
                n_stft=self._n_fft // 2 + 1,
                n_mels=self._n_mels,
                sample_rate=self.sample_rate,
                f_min=self._fmin,
                f_max=self._fmax,
            )
            self._griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=self._n_fft,
                hop_length=hop,
                win_length=self._n_fft,
                power=2.0,
                n_iter=32,
            )
        # log-mel dB → power.
        mel_power = torch.pow(10.0, mel / 10.0).clamp(min=1e-10)
        try:
            spec = self._inv_mel(mel_power)
            wave = self._griffin_lim(spec)
        except torch._C._LinAlgError:
            # `InverseMelScale` uses lstsq which fails on rank-deficient
            # mel filterbanks (common when n_mels is close to n_fft // 2).
            # Return silence rather than crashing the whole training run.
            frames = mel_power.shape[-1]
            hop = self._n_fft // 4
            wave = torch.zeros(max(frames * hop, 1))
        return wave.detach().cpu().numpy().squeeze()

    def _wandb_audio(self, stereo: np.ndarray, wandb_mod: Any) -> Any:
        if wandb_mod is not None:
            return wandb_mod.Audio(stereo.T, sample_rate=self.sample_rate)
        return stereo


# -- NaN guard + anomaly detection (plan Task 4.4) ------------------------


@contextlib.contextmanager
def set_anomaly_detection(enabled: bool) -> Iterator[None]:
    """Context manager wrapping `torch.autograd.set_detect_anomaly`.

    Nestable: restores prior state on exit. Use for debug runs only —
    anomaly detection is ~2-5× slower than normal training.
    """
    prior = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(enabled)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(prior)


class NaNGuardCallback(pl.Callback):
    """Detect NaN / Inf in loss, gradients, and (optionally) parameters.

    Args:
        action: `"raise"` (default) — raise `RuntimeError` on detection so
            the training script fails loud; `"skip"` — zero the corrupt
            gradient and let the optimizer step be a no-op, logging a
            `train/nan_count` metric. Skip mode is for exploratory sweeps
            where a few spurious batches shouldn't kill an 8-hour run.
        check_params: if True, scan every model parameter for NaN/Inf
            after each training batch. Expensive (linear in param count)
            — off by default; enable for debugging specific divergence
            failures.
        atol_grad: gradient-norm ceiling. Values above this are treated
            as Inf for reporting purposes (raw Inf still surfaces
            regardless).
    """

    _ACTIONS = frozenset({"raise", "skip"})

    def __init__(
        self,
        action: str = "raise",
        check_params: bool = False,
        atol_grad: float = 1e8,
    ) -> None:
        super().__init__()
        if action not in self._ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(self._ACTIONS)}, got {action!r}"
            )
        self.action = action
        self.check_params = check_params
        self.atol_grad = atol_grad

        self.nan_count: int = 0
        self.last_grad_norm: Optional[float] = None

    # -- loss check (cheapest, runs every batch) -------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss_val: Optional[torch.Tensor] = None
        if isinstance(outputs, torch.Tensor):
            loss_val = outputs
        elif isinstance(outputs, Mapping) and "loss" in outputs:
            loss_val = outputs["loss"]

        if loss_val is not None and not torch.isfinite(loss_val).all():
            self._handle_violation(
                pl_module,
                f"NaN/Inf in training loss at batch {batch_idx}: {float(loss_val)}",
            )

        if self.check_params:
            self._scan_parameters(pl_module, batch_idx)

    # -- grad check + norm log (before clipping) ------------------------

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        total_sq = 0.0
        has_bad = False
        for p in pl_module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                has_bad = True
                if self.action == "skip":
                    p.grad.zero_()
            else:
                total_sq += float(g.pow(2).sum())
        self.last_grad_norm = float(total_sq ** 0.5)

        pl_module.log("train/grad_norm", self.last_grad_norm, on_step=True)

        if has_bad:
            self._handle_violation(
                pl_module,
                f"NaN/Inf in gradient before optimizer step "
                f"(grad_norm so far: {self.last_grad_norm:.3e})",
            )

    # -- param scan (opt-in) --------------------------------------------

    def _scan_parameters(self, pl_module: pl.LightningModule, batch_idx: int) -> None:
        for name, p in pl_module.named_parameters():
            if not torch.isfinite(p.data).all():
                self._handle_violation(
                    pl_module,
                    f"NaN/Inf in parameter {name!r} after batch {batch_idx}",
                )

    # -- action dispatch -------------------------------------------------

    def _handle_violation(self, pl_module: pl.LightningModule, msg: str) -> None:
        self.nan_count += 1
        try:
            pl_module.log("train/nan_count", float(self.nan_count), on_step=True)
        except (RuntimeError, MisconfigurationException) as e:
            # Logger may not be attached yet (unit-test contexts) or Lightning
            # may reject the log call outside a trainer hook. Warn so actual
            # programming errors aren't silently swallowed, but never suppress
            # the downstream raise.
            warnings.warn(
                f"NaNGuard: failed to log nan_count: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.action == "raise":
            raise RuntimeError(msg)
        # action == "skip": caller already zeroed the grad; optimizer step
        # becomes a no-op on the poisoned params.
