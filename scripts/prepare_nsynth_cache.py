"""Pre-compute audio features (mel or CQT) for NSynth-bass and store to HDF5.

Rationale: Feature transforms are CPU-bound. Running per-item inside the
DataLoader idles the GPU at training time. One-time pre-compute caches
float16 features + label arrays for fast indexed reads by `NSynthBassCached`.

Usage (CQT — recommended):
    python scripts/prepare_nsynth_cache.py \
        --src $NSYNTH_ROOT --split train --out $NSYNTH_CACHE/train_cqt.h5 \
        --feature-type cqt

Usage (Mel — legacy):
    python scripts/prepare_nsynth_cache.py \
        --src $NSYNTH_ROOT --split train --out $NSYNTH_CACHE/train.h5 \
        --feature-type mel

CQT produces HDF5 with:
    cqt                 float16  (N, n_cqt_bins, T_frames)   [n_cqt_bins = 12*7 = 84]
    pitch, velocity, ...

Mel produces HDF5 with:
    mel                 float16  (N, n_mels, T_frames)       [n_mels = 128]
    pitch, velocity, ...

Attributes: sample_rate, duration, feature_type, hop_length, and feature-specific params.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.features import ConstantQ, LogMel
from src.data.nsynth import NSynthBass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True,
                   help="Root dir containing nsynth-{train,valid,test}/")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--out", type=Path, required=True,
                   help="HDF5 output path (parent dir will be created).")
    p.add_argument("--feature-type", choices=["mel", "cqt"], default="cqt",
                   help="Feature type: 'mel' (legacy) or 'cqt' (recommended).")
    
    # Shared params
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Number of waveforms per forward (GPU if available).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Mel-specific params
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=400)
    
    # CQT-specific params
    p.add_argument("--n-bins-per-octave", type=int, default=12)
    p.add_argument("--n-octaves", type=int, default=7)
    p.add_argument("--fmin", type=float, default=32.7)
    
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = NSynthBass(
        root=args.src,
        split=args.split,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"NSynthBass({args.split}) returned 0 items — check --src")

    # Instantiate feature extractor
    if args.feature_type == "cqt":
        feature_extractor = ConstantQ(
            sample_rate=args.sample_rate,
            n_bins_per_octave=args.n_bins_per_octave,
            n_octaves=args.n_octaves,
            fmin=args.fmin,
            hop_length=args.hop_length,
        ).to(args.device)
        n_bins = args.n_bins_per_octave * args.n_octaves
        feature_key = "cqt"
        print(f"[prepare] Feature type: CQT (n_bins={n_bins})")
    else:  # mel
        feature_extractor = LogMel(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        ).to(args.device)
        n_bins = args.n_mels
        feature_key = "mel"
        print(f"[prepare] Feature type: Mel (n_mels={args.n_mels})")

    feature_extractor.train(False)

    # One dry forward to learn T_frames for dataset allocation.
    wav0, _ = ds[0]
    with torch.no_grad():
        probe = feature_extractor(wav0.unsqueeze(0).to(args.device))  # (1, 1, n_bins, T)
    t_frames = int(probe.shape[-1])
    print(f"[prepare] N={n}  n_bins={n_bins}  T_frames={t_frames}  out={args.out}")

    # Preallocate HDF5 datasets.
    with h5py.File(args.out, "w") as f:
        # Attributes
        f.attrs["sample_rate"] = args.sample_rate
        f.attrs["duration"] = args.duration
        f.attrs["hop_length"] = args.hop_length
        f.attrs["split"] = args.split
        f.attrs["feature_type"] = args.feature_type

        if args.feature_type == "cqt":
            f.attrs["n_bins_per_octave"] = args.n_bins_per_octave
            f.attrs["n_octaves"] = args.n_octaves
            f.attrs["fmin"] = args.fmin
        else:
            f.attrs["n_mels"] = args.n_mels
            f.attrs["n_fft"] = args.n_fft

        # Feature dataset (main output)
        feature_ds = f.create_dataset(
            feature_key,
            shape=(n, n_bins, t_frames),
            dtype=np.float16,
            chunks=(1, n_bins, t_frames),
            compression="lzf",
        )

        # Label datasets
        pitch_ds = f.create_dataset("pitch", shape=(n,), dtype=np.int16)
        velocity_ds = f.create_dataset("velocity", shape=(n,), dtype=np.int16)
        isrc_ds = f.create_dataset("instrument_source", shape=(n,), dtype=np.int16)
        iid_ds = f.create_dataset("instrument_id", shape=(n,), dtype=np.int16)

        t0 = time.perf_counter()
        bs = args.batch_size
        for start in tqdm(range(0, n, bs), desc=f"[{args.split}] {feature_key}-cache"):
            end = min(start + bs, n)
            wavs: list[torch.Tensor] = []
            pits: list[int] = []
            vels: list[int] = []
            isrcs: list[int] = []
            iids: list[int] = []
            for i in range(start, end):
                wav, lbl = ds[i]
                wavs.append(wav)  # (1, T)
                pits.append(int(lbl["pitch"]))
                vels.append(int(lbl["velocity"]))
                isrcs.append(int(lbl["instrument_source"]))
                iids.append(int(lbl["instrument_id"]))

            batch = torch.stack(wavs, dim=0).to(args.device, non_blocking=True)  # (B, 1, T)
            with torch.no_grad():
                out = feature_extractor(batch)  # (B, 1, n_bins, T)
            out = out.squeeze(1).to(torch.float16).cpu().numpy()
            feature_ds[start:end] = out
            pitch_ds[start:end] = np.array(pits, dtype=np.int16)
            velocity_ds[start:end] = np.array(vels, dtype=np.int16)
            isrc_ds[start:end] = np.array(isrcs, dtype=np.int16)
            iid_ds[start:end] = np.array(iids, dtype=np.int16)

        dt = time.perf_counter() - t0
    
    size_mb = args.out.stat().st_size / (1024 ** 2)
    print(f"[done] wrote {args.out}  ({size_mb:.1f} MB) in {dt:.1f}s")


if __name__ == "__main__":
    main()
