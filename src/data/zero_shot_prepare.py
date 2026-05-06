"""Offline preparation of zero-shot OOD bass shards.

Given a directory of full-length mono bass stems (one `.wav` per song), this
module segments each stem into overlapping windows, extracts a CREPE pitch
curve (Kim et al. 2018) + confidence per frame, and produces `.pt` shards
consumed by `src.data.zero_shot.ZeroShotBass`.

Design notes
------------
- CREPE runs once on the full stem, not per segment, so neighboring segments
  see a coherent pitch curve (avoids window-edge artifacts).
- Beat grid comes from `librosa.beat.beat_track` on the full stem; per-segment
  beat timestamps are the subset that falls inside the window, re-origined so
  `beat_grid[i]` is seconds from the segment start.
- Shards are one `.pt` per segment — keeps `__getitem__` trivially lazy and
  parallelizable across DataLoader workers.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torchaudio
import torchcrepe

from src.utils.device import resolve_device


_AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")
# CREPE accepts 16 kHz inputs natively (model is trained at 16 kHz).
_CREPE_SR: int = 16000
# CREPE bass-friendly pitch range. Lower bound pinned to C1 (32.70 Hz) — CREPE's
# internal cent-bin grid starts there; setting fmin below C1 collapses the
# periodicity softmax and returns log(0) = -inf for every frame (torchcrepe
# 0.0.23). C1 covers everything above 5-string-bass low B (B0 = 30.87 Hz is
# one semitone below and will snap to C1; standard 4-string bass is E1 = 41 Hz).
_CREPE_FMIN: float = 32.70
_CREPE_FMAX: float = 660.0


def _list_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _AUDIO_EXTS)


def _load_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(torch.float32, copy=False)


def _crepe_pitch(
    waveform: torch.Tensor,
    sample_rate: int,
    step_ms: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict (pitch_hz, confidence) curves at `step_ms` intervals.

    Runs on the supplied device (CUDA preferred; MPS supported when a build of
    torchcrepe is compatible; falls back to CPU). Results are returned on CPU
    so downstream torch.save serializes cleanly across nodes.
    """
    if sample_rate != _CREPE_SR:
        # torchcrepe expects 16 kHz; resample-once is the documented path.
        waveform = torchaudio.functional.resample(waveform, sample_rate, _CREPE_SR)
    hop_length = int(_CREPE_SR * step_ms / 1000.0)
    # torchcrepe uses string device names ("cuda", "cpu"); it does not yet
    # handle "mps" — fall back to CPU when MPS is requested. Surface the
    # downgrade so users on Apple-Silicon dev boxes know why CREPE is slow.
    if device.type in {"cuda", "cpu"}:
        dev_str = device.type
    else:
        warnings.warn(
            f"torchcrepe 0.0.23 does not support device '{device.type}'; "
            "falling back to CPU (CREPE inference will be significantly slower)",
            RuntimeWarning,
            stacklevel=2,
        )
        dev_str = "cpu"
    # torchcrepe.predict returns (pitch[1, T], periodicity[1, T]).
    pitch, confidence = torchcrepe.predict(
        waveform.to(dev_str),
        sample_rate=_CREPE_SR,
        hop_length=hop_length,
        fmin=_CREPE_FMIN,
        fmax=_CREPE_FMAX,
        model="tiny",  # the 'tiny' variant is accurate for bass + 10× faster than 'full'
        return_periodicity=True,
        device=dev_str,
        batch_size=1024,
        pad=True,
    )
    return pitch.squeeze(0).cpu().to(torch.float32), confidence.squeeze(0).cpu().to(torch.float32)


def _beat_grid(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Return beat times in seconds (ascending, 1-D np.ndarray). May be empty."""
    x = waveform.detach().cpu().squeeze(0).numpy()
    try:
        _, beats = librosa.beat.beat_track(y=x, sr=sample_rate, units="time")
    except (ValueError, librosa.util.exceptions.ParameterError):
        # Short / near-silent stems raise inside librosa. An empty grid is a
        # valid output — downstream code handles it. Narrow the catch so
        # MemoryError / KeyboardInterrupt / RuntimeError propagate rather
        # than silently zero-ing out the beat grid on SLURM OOM.
        return np.zeros(0, dtype=np.float32)
    return np.asarray(beats, dtype=np.float32)


def _segment_spans(
    total_samples: int,
    segment_samples: int,
    hop_samples: int,
) -> list[tuple[int, int]]:
    """Emit (start, end) sample spans for complete segments only.

    Partial tail segments are dropped — the zero-shot set treats each window as
    a self-contained probe, padding would bias pitch statistics.
    """
    if total_samples < segment_samples:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start + segment_samples <= total_samples:
        spans.append((start, start + segment_samples))
        start += hop_samples
    return spans


def prepare_zero_shot_shards(
    input_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int = 16000,
    segment_seconds: float = 4.0,
    hop_seconds: float = 2.0,
    crepe_step_ms: float = 10.0,
    device: str | torch.device | None = "auto",
) -> list[Path]:
    """Segment stems + extract CREPE pitch/beats; write one `.pt` per segment.

    Args:
        device: `"auto"` (default) → CUDA → MPS → CPU; or pass an explicit
            torch device. MPS is auto-downgraded to CPU for the CREPE call
            because torchcrepe 0.0.23 does not yet support MPS.

    Returns the list of shard paths written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = resolve_device(device)

    segment_samples = int(sample_rate * segment_seconds)
    hop_samples = int(sample_rate * hop_seconds)
    frames_per_second = int(round(1000.0 / crepe_step_ms))
    frames_per_segment = int(round(segment_seconds * frames_per_second))

    shard_paths: list[Path] = []

    for wav_path in _list_inputs(input_dir):
        track_name = wav_path.stem
        waveform = _load_mono(wav_path, sample_rate)
        total = waveform.size(-1)

        pitch_full, conf_full = _crepe_pitch(waveform, sample_rate, crepe_step_ms, device_t)
        beats_full = _beat_grid(waveform, sample_rate)

        for seg_idx, (s, e) in enumerate(_segment_spans(total, segment_samples, hop_samples)):
            seg_wav = waveform[..., s:e].clone()

            # Frame-align pitch/confidence slices. CREPE frames are centered at
            # `i * hop + hop/2`; use an inclusive start → start+frames_per_segment.
            frame_start = int(round(s * frames_per_second / sample_rate))
            frame_end = frame_start + frames_per_segment
            pitch = pitch_full[frame_start:frame_end]
            conf = conf_full[frame_start:frame_end]
            if pitch.numel() < frames_per_segment:
                # Right-pad with zeros if CREPE produced one fewer frame than
                # expected at the last window.
                pad = frames_per_segment - pitch.numel()
                pitch = torch.nn.functional.pad(pitch, (0, pad))
                conf = torch.nn.functional.pad(conf, (0, pad))

            seg_start_s = s / sample_rate
            seg_end_s = e / sample_rate
            mask = (beats_full >= seg_start_s) & (beats_full < seg_end_s)
            beat_times = beats_full[mask] - seg_start_s

            shard = {
                "waveform": seg_wav,                  # (1, segment_samples) float32
                "track_name": track_name,
                "segment_idx": seg_idx,
                "sample_rate": sample_rate,
                "crepe_pitch_curve": pitch,           # (frames_per_segment,) Hz
                "crepe_confidence": conf,             # (frames_per_segment,) [0, 1]
                "beat_grid": torch.from_numpy(beat_times).to(torch.float32),
            }
            shard_path = output_dir / f"{track_name}-{seg_idx:04d}.pt"
            torch.save(shard, shard_path)
            shard_paths.append(shard_path)

    return shard_paths


def _iter_shards(shard_dir: Path) -> Iterable[Path]:
    yield from sorted(shard_dir.glob("*.pt"))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Segment zero-shot bass stems and extract CREPE pitch curves + beats.",
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--hop-seconds", type=float, default=2.0)
    p.add_argument("--crepe-step-ms", type=float, default=10.0)
    p.add_argument("--device", type=str, default="auto",
                   help='"auto" | "cuda" | "mps" | "cpu"')
    args = p.parse_args()

    paths = prepare_zero_shot_shards(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        hop_seconds=args.hop_seconds,
        crepe_step_ms=args.crepe_step_ms,
        device=args.device,
    )
    print(f"wrote {len(paths)} shards to {args.output_dir}")


if __name__ == "__main__":
    main()
