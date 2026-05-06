"""Synthetic pre-training dataset for SC-VAE (plan Task 7.3).

Reference: "Self-Supervised Causal Representation Learning with Synthetic
Data" (arXiv:2505.23305) — simulation-as-supervision for causal factor
recovery.

Each item:
    x          — (1, num_samples) mono waveform
    pitch_hz   — continuous fundamental (bass range)
    timbre_id  — categorical over `n_timbres` presets (filter + ADSR +
                 oscillator mix)

Generation is fully deterministic given the dataset seed and the item
index, so the same index yields the same item across workers.

Design note: this is a pre-training dataset only — the goal is simulation-
to-real transfer, not audio realism. Tone is spectrally rich (sine +
sawtooth + filtered-noise tail), loudness-shaped by an ADSR envelope, and
timbrally separated by biquad low-pass + formant parameters drawn from a
fixed preset table.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Preset table — each row = (cutoff_hz, q, adsr, osc_mix)


def _default_presets(n_timbres: int) -> list[dict[str, Any]]:
    """Generate `n_timbres` presets spanning cutoff x Q x ADSR x mix."""
    rng = np.random.default_rng(seed=12345)
    presets: list[dict[str, Any]] = []
    for i in range(n_timbres):
        cutoff = float(rng.uniform(150.0, 1800.0))
        q = float(rng.uniform(0.5, 4.0))
        attack = float(rng.uniform(0.001, 0.05))
        decay = float(rng.uniform(0.02, 0.2))
        sustain = float(rng.uniform(0.3, 0.9))
        release = float(rng.uniform(0.05, 0.3))
        saw_mix = float(rng.uniform(0.0, 1.0))  # 0 = pure sine, 1 = pure saw
        noise_gain = float(rng.uniform(0.0, 0.05))
        presets.append({
            "cutoff": cutoff, "q": q,
            "attack": attack, "decay": decay,
            "sustain": sustain, "release": release,
            "saw_mix": saw_mix, "noise_gain": noise_gain,
        })
    return presets


# ---------------------------------------------------------------------------
# Oscillators / ADSR / biquad low-pass


def _sine_osc(freq: float, n_samples: int, sample_rate: float) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    return np.sin(2.0 * math.pi * freq * t)


def _sawtooth_osc(freq: float, n_samples: int, sample_rate: float) -> np.ndarray:
    """Bandlimited additive saw via the first K harmonics below Nyquist."""
    nyquist = 0.5 * sample_rate
    max_k = max(1, int(nyquist / max(freq, 1e-6)))
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    out = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, max_k + 1):
        out += (1.0 / k) * np.sin(2.0 * math.pi * k * freq * t)
    # Normalize amplitude to [-1, 1] — additive saw peak ~ ln(K) + 0.577.
    return (2.0 / math.pi) * out


def _adsr_envelope(
    n_samples: int,
    sample_rate: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    """Standard ADSR. Release occupies the final `release` seconds."""
    env = np.zeros(n_samples, dtype=np.float64)
    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)
    # Compress stages proportionally when they exceed n_samples so short
    # crops (0.25 s) still get all three phases represented.
    total = a + d + r
    if total > n_samples:
        scale = n_samples / max(total, 1)
        a = int(a * scale)
        d = int(d * scale)
        r = max(0, n_samples - a - d)
    s_len = max(0, n_samples - a - d - r)
    idx = 0
    if a > 0:
        env[idx:idx + a] = np.linspace(0.0, 1.0, a, endpoint=False)
        idx += a
    if d > 0:
        env[idx:idx + d] = np.linspace(1.0, sustain, d, endpoint=False)
        idx += d
    if s_len > 0:
        env[idx:idx + s_len] = sustain
        idx += s_len
    if r > 0:
        end = min(idx + r, n_samples)
        env[idx:end] = np.linspace(sustain, 0.0, end - idx, endpoint=True)
    return env


def _biquad_lowpass(
    x: np.ndarray, cutoff: float, q: float, sample_rate: float
) -> np.ndarray:
    """Low-pass biquad (RBJ cookbook) via scipy.signal.lfilter — C loop
    instead of a Python-level per-sample recurrence (DataLoader workers
    would bottleneck on 64k-sample crops otherwise).
    """
    from scipy.signal import lfilter

    w0 = 2.0 * math.pi * cutoff / sample_rate
    alpha = math.sin(w0) / (2.0 * max(q, 1e-6))
    cos_w0 = math.cos(w0)

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, a, x)


# ---------------------------------------------------------------------------
# Dataset


class SynthBassPretrainDataset(Dataset):
    """On-the-fly synthesized bass-ish tones with ground-truth factors.

    Args:
        n: items per epoch.
        sample_rate: output sample rate (Hz).
        crop_seconds: output duration (seconds).
        n_timbres: number of fixed timbre presets (integer factor support).
        pitch_hz_range: continuous fundamental range `(f_min, f_max)`.
        seed: base RNG seed for reproducibility.
    """

    def __init__(
        self,
        *,
        n: int = 10_000,
        sample_rate: int = 16000,
        crop_seconds: float = 4.0,
        n_timbres: int = 16,
        pitch_hz_range: tuple[float, float] = (40.0, 300.0),
        seed: int = 0,
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if crop_seconds <= 0:
            raise ValueError(f"crop_seconds must be > 0, got {crop_seconds}")
        if n_timbres < 1:
            raise ValueError(f"n_timbres must be >= 1, got {n_timbres}")
        lo, hi = pitch_hz_range
        if lo <= 0.0 or hi < lo:
            raise ValueError(
                f"pitch_hz_range must be (lo>0, hi>=lo), got {pitch_hz_range}"
            )

        self.n = int(n)
        self.sample_rate = int(sample_rate)
        self.num_samples = int(round(sample_rate * crop_seconds))
        self.crop_seconds = float(crop_seconds)
        self.n_timbres = int(n_timbres)
        self.pitch_hz_range = (float(lo), float(hi))
        self.seed = int(seed)
        self._presets = _default_presets(self.n_timbres)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng((self.seed, idx))
        lo, hi = self.pitch_hz_range
        pitch_hz = float(rng.uniform(lo, hi))
        timbre_id = int(rng.integers(0, self.n_timbres))
        preset = self._presets[timbre_id]

        sine = _sine_osc(pitch_hz, self.num_samples, self.sample_rate)
        saw = _sawtooth_osc(pitch_hz, self.num_samples, self.sample_rate)
        osc = (1.0 - preset["saw_mix"]) * sine + preset["saw_mix"] * saw

        if preset["noise_gain"] > 0.0:
            noise = rng.standard_normal(self.num_samples)
            osc = osc + preset["noise_gain"] * noise

        env = _adsr_envelope(
            n_samples=self.num_samples,
            sample_rate=self.sample_rate,
            attack=preset["attack"],
            decay=preset["decay"],
            sustain=preset["sustain"],
            release=preset["release"],
        )

        filtered = _biquad_lowpass(
            osc * env,
            cutoff=preset["cutoff"],
            q=preset["q"],
            sample_rate=float(self.sample_rate),
        )

        # Peak-normalize to avoid clipping in downstream loaders.
        peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
        if peak > 1e-6:
            filtered = filtered / peak

        x = torch.from_numpy(filtered.astype(np.float32)).unsqueeze(0)  # (1, T)
        return {
            "x": x,
            "labels": {"pitch_hz": pitch_hz, "timbre_id": timbre_id},
        }
