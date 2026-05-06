"""Pre-compute and cache CREPE f0 for NSynth-bass source clips.

Run once before evaluation. Source-clip f0 is constant across checkpoints,
so caching it avoids re-running CREPE inference (~1 h on L40) per eval call.

Output HDF5 schema:
    f0      (N, F)  float32  — f0 in Hz per frame
    voiced  (N, F)  bool     — voiced mask (periodicity >= threshold)
    attrs:
        sample_rate    int
        hop_length     int
        n_clips        int

Usage:
    python scripts/prepare_crepe_cache.py \
        --src $NSYNTH_ROOT \
        --split train \
        --out $CREPE_CACHE/train_f0.h5 \
        --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.nsynth import NSynthBass
from src.evaluation.swap_pitch_acc import crepe_extractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache CREPE f0 for NSynth-bass clips.")
    p.add_argument("--src", type=Path, required=True, help="NSynth root dir containing nsynth-{train,valid,test}/")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--out", type=Path, required=True, help="Output HDF5 path.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", type=str, default="full", choices=["tiny", "small", "medium", "large", "full"])
    p.add_argument("--periodicity-threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dataset = NSynthBass(root=args.src, split=args.split)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"NSynthBass({args.split!r}) returned 0 items — check --src")

    sample_rate: int = dataset.sample_rate  # type: ignore[attr-defined]
    hop_length = max(1, int(round(sample_rate * 0.01)))

    # Probe first item to get frame count F.
    probe_wav, _ = dataset[0]
    # probe_wav: (1, T) — convert to (1, T) mono batch
    probe_wav_batch = probe_wav[:1].to(args.device)  # (1, T)
    if probe_wav_batch.dim() == 2 and probe_wav_batch.shape[0] == 1:
        pass  # already (1, T)
    elif probe_wav_batch.dim() == 3:
        probe_wav_batch = probe_wav_batch.squeeze(0)  # (1, T)

    f0_probe, _ = crepe_extractor(
        probe_wav_batch,
        sample_rate=sample_rate,
        hop_length=hop_length,
        model=args.model,
        device=args.device,
        periodicity_threshold=args.periodicity_threshold,
    )
    n_frames = f0_probe.shape[-1]

    print(f"[prepare_crepe] N={n}  sample_rate={sample_rate}  hop={hop_length}  frames={n_frames}")
    print(f"[prepare_crepe] Device={args.device}  CREPE model={args.model}")
    print(f"[prepare_crepe] Output → {args.out}")

    with h5py.File(args.out, "w") as hf:
        f0_ds = hf.create_dataset("f0", shape=(n, n_frames), dtype=np.float32)
        voiced_ds = hf.create_dataset("voiced", shape=(n, n_frames), dtype=bool)
        hf.attrs["sample_rate"] = sample_rate
        hf.attrs["hop_length"] = hop_length
        hf.attrs["n_clips"] = n
        hf.attrs["split"] = args.split

        for start in tqdm(range(0, n, args.batch_size), desc=f"[{args.split}] crepe-cache"):
            end = min(start + args.batch_size, n)
            wavs = []
            for i in range(start, end):
                wav, _ = dataset[i]
                # wav: (1, T) or (T,) — normalise to (T,)
                if wav.dim() == 2:
                    wav = wav.squeeze(0)
                wavs.append(wav)

            # Stack to (B, T)
            wav_batch = torch.stack(wavs, dim=0).to(args.device)

            with torch.no_grad():
                f0_batch, voiced_batch = crepe_extractor(
                    wav_batch,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    model=args.model,
                    device=args.device,
                    periodicity_threshold=args.periodicity_threshold,
                )

            f0_ds[start:end] = f0_batch.cpu().numpy().astype(np.float32)
            voiced_ds[start:end] = voiced_batch.cpu().numpy()

    size_mb = args.out.stat().st_size / (1024 ** 2)
    print(f"[done] wrote {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
