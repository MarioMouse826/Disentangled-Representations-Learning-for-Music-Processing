"""NSynth bass-family loader.

Reference: Engel et al. 2017, "Neural Audio Synthesis of Musical Notes with WaveNet
Autoencoders" (arXiv:1704.01279). We restrict to `instrument_family_str == "bass"`.

NSynth on-disk layout expected by this loader:

    <root>/nsynth-<split>/examples.json
    <root>/nsynth-<split>/audio/<note_str>.wav

`examples.json` is a dict keyed by `note_str` (no trailing extension). Each value
contains at minimum `pitch`, `velocity`, `instrument_family_str`,
`instrument_source_str`, `instrument`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, TypedDict

import torch
import torchaudio
from torch.utils.data import Dataset


class NSynthLabel(TypedDict):
    pitch: torch.Tensor
    velocity: torch.Tensor
    instrument_source: torch.Tensor
    instrument_id: torch.Tensor


class NSynthBass(Dataset):
    """Bass-family subset of NSynth.

    Returns `(waveform: Tensor[1, T], label: NSynthLabel)` where
    `T = int(sample_rate * duration)`. Waveforms are float32, mono, either
    truncated or zero-padded to exactly `T` samples.

    Args:
        root: directory containing `nsynth-<split>/` subfolders.
        split: one of `"train"`, `"valid"`, `"test"`.
        sample_rate: target sample rate. Source audio is resampled if it differs.
        duration: clip length in seconds.
        transform: optional callable `Tensor -> Tensor` applied after length-fix.
    """

    SOURCES: dict[str, int] = {"acoustic": 0, "electronic": 1, "synthetic": 2}
    _SPLITS: frozenset[str] = frozenset({"train", "valid", "test"})
    _PITCH_MIN: int = 0    # Allow extended MIDI range for bass instruments
    _PITCH_MAX: int = 127  # Full MIDI range (was 21-108, too restrictive for bass)

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        duration: float = 4.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {sorted(self._SPLITS)}, got {split!r}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        self.root: Path = Path(root) / f"nsynth-{split}"
        self.audio_dir: Path = self.root / "audio"
        manifest_path: Path = self.root / "examples.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")

        self.sample_rate: int = sample_rate
        self.num_samples: int = int(sample_rate * duration)
        self.transform = transform
        self._items: list[tuple[str, dict]] = self._load_manifest(manifest_path)

    # Index of (key, precomputed-label-dict). We pre-decode label fields once at
    # __init__ so __getitem__ stays hot-path I/O + tensor construction only.
    def _load_manifest(self, path: Path) -> list[tuple[str, dict]]:
        with open(path) as f:
            raw = json.load(f)

        items: list[tuple[str, dict]] = []
        for key, meta in raw.items():
            if meta.get("instrument_family_str") != "bass":
                continue
            pitch = int(meta["pitch"])
            if not self._PITCH_MIN <= pitch <= self._PITCH_MAX:
                raise ValueError(
                    f"pitch out of MIDI range [{self._PITCH_MIN}, {self._PITCH_MAX}] "
                    f"for note {key!r}: got {pitch}"
                )
            source_str = meta["instrument_source_str"]
            if source_str not in self.SOURCES:
                raise ValueError(
                    f"unknown instrument_source_str for {key!r}: {source_str!r}"
                )
            items.append(
                (
                    key,
                    {
                        "pitch": pitch,
                        "velocity": int(meta["velocity"]),
                        "instrument_source": self.SOURCES[source_str],
                        "instrument_id": int(meta["instrument"]),
                    },
                )
            )

        # Deterministic ordering — required so that `(seed, split) -> example[i]`
        # is reproducible across machines and Python versions (dict insertion
        # order from json.load is not guaranteed equivalent across environments).
        items.sort(key=lambda kv: kv[0])
        return items

    def __len__(self) -> int:
        return len(self._items)

    def _load_audio(self, key: str) -> torch.Tensor:
        path = self.audio_dir / f"{key}.wav"
        wav, sr = torchaudio.load(str(path))  # (channels, frames)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Force mono. NSynth is mono but guard against preprocessing drift.
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Length-fix: truncate or right-pad with zeros to exactly num_samples.
        length = wav.size(-1)
        if length > self.num_samples:
            wav = wav[..., : self.num_samples]
        elif length < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - length))

        return wav.to(torch.float32, copy=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, NSynthLabel]:
        key, meta = self._items[idx]
        wav = self._load_audio(key)
        if self.transform is not None:
            wav = self.transform(wav)
        label: NSynthLabel = {
            "pitch": torch.tensor(meta["pitch"], dtype=torch.long),
            "velocity": torch.tensor(meta["velocity"], dtype=torch.long),
            "instrument_source": torch.tensor(meta["instrument_source"], dtype=torch.long),
            "instrument_id": torch.tensor(meta["instrument_id"], dtype=torch.long),
        }
        return wav, label
