"""Timbre-swap pitch accuracy — task-based perceptual eval.

Pati, A. & Lerch, A., 2021. "Is Disentanglement Enough? On Latent
Representations for Controllable Music Generation." arXiv:2108.01450.

Protocol (per pair (x_a, x_b)):
    1. Encode both; decode with z_s from x_b, z_c from x_a.
    2. Vocoder to waveform x_hat.
    3. Extract f0(x_hat) via CREPE (torchcrepe).
    4. PitchAcc = fraction of voiced frames within `tolerance_cents` of f0(x_a).

Aggregate mean across pairs with bootstrap CI95.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Frame-level primitives


def cents_error(
    pred_hz: torch.Tensor, ref_hz: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Absolute cents difference `|1200 * log2(pred / ref)|` per frame.

    Unvoiced frames (where either pred or ref is <= eps) are mapped to 0.0.
    """
    if pred_hz.shape != ref_hz.shape:
        raise ValueError(
            f"pred/ref shape mismatch: {tuple(pred_hz.shape)} vs "
            f"{tuple(ref_hz.shape)}"
        )
    mask = (pred_hz > eps) & (ref_hz > eps)
    ratio = torch.where(mask, pred_hz / ref_hz.clamp_min(eps), torch.ones_like(pred_hz))
    cents = 1200.0 * torch.log2(ratio.clamp_min(eps))
    cents = cents.abs()
    return torch.where(mask, cents, torch.zeros_like(cents))


def frame_pitch_accuracy(
    pred_hz: torch.Tensor,
    ref_hz: torch.Tensor,
    *,
    voiced: torch.Tensor,
    tolerance_cents: float = 50.0,
) -> float:
    """Fraction of voiced frames whose cents error is within tolerance."""
    if not voiced.any():
        raise ValueError("no voiced frames in reference")
    cents = cents_error(pred_hz, ref_hz)
    correct = (cents <= tolerance_cents) & voiced
    return float(correct.sum()) / float(voiced.sum())


# ---------------------------------------------------------------------------
# Bootstrap


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> dict[str, float]:
    """Percentile-bootstrap mean + (1 - alpha) CI."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("cannot bootstrap empty array")
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = v.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = v[idx].mean()
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# CREPE adapter


def crepe_extractor(
    wave: torch.Tensor,
    sample_rate: int,
    *,
    hop_length: int | None = None,
    fmin: float = 50.0,
    fmax: float = 2006.0,
    model: str = "full",
    device: str | torch.device = "cpu",
    periodicity_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run torchcrepe on a batch of mono waveforms.

    Args:
        wave: `(B, T)` float tensor.
        hop_length: defaults to 10 ms at the given sample_rate.

    Returns:
        `(f0_hz, voiced_mask)` shaped `(B, F)` each. Voiced = periodicity
        above `periodicity_threshold`.
    """
    import torchcrepe

    if wave.ndim != 2:
        raise ValueError(f"wave must be (B, T), got shape {tuple(wave.shape)}")
    if hop_length is None:
        hop_length = max(1, int(round(sample_rate * 0.01)))

    f0, periodicity = torchcrepe.predict(
        wave,
        sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        return_periodicity=True,
        device=str(device),
    )
    voiced = periodicity >= periodicity_threshold
    return f0, voiced


# ---------------------------------------------------------------------------
# End-to-end protocol


class _ModelLike(Protocol):
    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...
    def decode(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor: ...


VocoderFn = Callable[[torch.Tensor], torch.Tensor]
PitchExtractorFn = Callable[
    [torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]


@torch.no_grad()
def compute_swap_pitch_accuracy(
    *,
    model: _ModelLike,
    vocoder: VocoderFn,
    pairs: Iterable[dict[str, Any]],
    pitch_extractor: PitchExtractorFn,
    sample_rate: int,
    tolerance_cents: float = 50.0,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Aggregate timbre-swap pitch accuracy over a list of pairs.

    Each pair dict:
        "x_a": mel of the pitch source
        "x_b": mel of the timbre source
        "f0_a": reference f0 Hz tensor (F,)
        "voiced_a": boolean mask (F,)

    Returns:
        {
            "mean_accuracy": float,
            "ci95_low": float,
            "ci95_high": float,
            "per_pair": (P,) ndarray,
        }
    """
    per_pair: list[float] = []
    for pair in pairs:
        x_a = pair["x_a"]
        x_b = pair["x_b"]
        ref_f0 = pair["f0_a"]
        voiced = pair["voiced_a"]

        enc_a = model.encode(x_a)
        enc_b = model.encode(x_b)
        mel_hat = model.decode(enc_b["mu_s"], enc_a["mu_c"])
        wave_hat = vocoder(mel_hat)  # (1, T)
        f0_hat, voiced_hat = pitch_extractor(wave_hat, sample_rate)

        # Align on the shorter frame count — CREPE + reference may differ by
        # a few frames depending on padding.
        f = min(f0_hat.shape[-1], ref_f0.shape[-1], voiced.shape[-1])
        pred = f0_hat[0, :f]
        ref = ref_f0[:f]
        vmask = voiced[:f] & voiced_hat[0, :f]
        if not vmask.any():
            # Degenerate pair — no overlapping voiced frames. Record 0.0.
            per_pair.append(0.0)
            continue
        acc = frame_pitch_accuracy(
            pred, ref, voiced=vmask, tolerance_cents=tolerance_cents
        )
        per_pair.append(acc)

    arr = np.asarray(per_pair, dtype=np.float64)
    ci = bootstrap_ci(arr, n_boot=n_boot, alpha=alpha, random_state=random_state)
    return {
        "mean_accuracy": ci["mean"],
        "ci95_low": ci["ci_low"],
        "ci95_high": ci["ci_high"],
        "per_pair": arr,
    }
