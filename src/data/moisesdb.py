"""MoisesDB bass-stem loader.

Reference: Pereira et al. 2024, "MoisesDB: A Dataset for Source Separation Beyond
4-Stems". We use the `bass` stem subfolder of each track. Tracks with no bass
material are skipped. Sub-stems within a track's `bass/` folder (e.g. DI +
amped takes) are summed into a single bass signal before downstream processing.

On-disk layout expected:

    <root>/<track_id>/data.json
    <root>/<track_id>/bass/<n>.wav           (0..N files; all summed)
    <root>/<track_id>/<other_stem>/...       (ignored)

Each dataset item is a segment of a bass signal. In `deterministic=True` mode,
segments are produced at a fixed hop; this is used for validation / test.
In `deterministic=False` mode, each `__getitem__` draws a random crop from the
full stem using a seeded RNG — used for training.

Both modes emit `(wav[1, T], label)` where
`T = int(sample_rate * crop_seconds)` and `label` is a `MoisesDBLabel` dict.

Splits: MoisesDB has no official split. We deterministically bucket tracks by
`hash(track_id)` into train / valid / test at 80 / 10 / 10 ratio. This keeps
splits stable across runs without a separate split-file dependency.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio


class MoisesDBLabel(TypedDict):
    track_id: str
    track_idx: int
    stem: str
    segment_idx: int


class MoisesDBBass(torch.utils.data.Dataset):
    """Bass-stem segment dataset backed by MoisesDB-layout audio.

    Args:
        root: directory whose immediate children are per-track folders.
        split: `"train"`, `"valid"`, `"test"`, or `"all"` (no filtering).
        crop_seconds: output segment length.
        sample_rate: target sample rate (source audio is resampled if different).
        deterministic: if True, segments are stride-indexed by `hop_seconds`;
            if False, each `__getitem__` returns one random crop per track.
        hop_seconds: stride between consecutive deterministic segments. Ignored
            when `deterministic=False`.
        target_lufs: integrated-loudness target in LUFS. If `None`, skip
            loudness normalization. Default −23 LUFS (EBU R128).
        seed: base RNG seed for non-deterministic crop sampling.
    """

    _SPLITS: frozenset[str] = frozenset({"train", "valid", "test", "all"})
    _BASS_STEM: str = "bass"
    _AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")

    # Split bucketing fractions — cumulative.
    _TRAIN_FRAC: float = 0.80
    _VALID_FRAC: float = 0.90  # remaining 0.10 → test

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_seconds: float = 4.0,
        sample_rate: int = 16000,
        deterministic: bool = False,
        hop_seconds: float = 2.0,
        target_lufs: float | None = -23.0,
        seed: int = 0,
    ) -> None:
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {sorted(self._SPLITS)}, got {split!r}")
        if crop_seconds <= 0:
            raise ValueError(f"crop_seconds must be positive, got {crop_seconds}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if hop_seconds <= 0:
            raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

        self.root: Path = Path(root)
        self.split: str = split
        self.sample_rate: int = sample_rate
        self.num_samples: int = int(sample_rate * crop_seconds)
        self.hop_samples: int = int(sample_rate * hop_seconds)
        self.deterministic: bool = deterministic
        self.target_lufs: float | None = target_lufs
        self.seed: int = seed

        tracks = self._discover_tracks()
        if not tracks:
            raise FileNotFoundError(f"no MoisesDB tracks with bass stems under {self.root}")

        tracks = [t for t in tracks if self._in_split(t["track_id"], split)]

        # Pre-load all bass stems into memory here because MoisesDB is small
        # (~240 tracks, a few minutes of bass audio per track after mono+16kHz
        # downsampling → under 2 GB RAM). Decoding on every __getitem__ would
        # dominate training step time — this trade keeps GPU fed.
        self._cache: list[dict] = [self._prepare_track(t) for t in tracks]

        # Index: list of (cache_idx, start_sample, segment_idx_within_track).
        # For deterministic mode, segment_idx counts hops; for random mode,
        # segment_idx is always 0 (one "slot" per track that re-samples).
        self._index: list[tuple[int, int, int]] = self._build_index()

    # -- track discovery --------------------------------------------------

    def _discover_tracks(self) -> list[dict]:
        tracks: list[dict] = []
        if not self.root.is_dir():
            return tracks
        for track_dir in sorted(self.root.iterdir()):
            if not track_dir.is_dir():
                continue
            bass_dir = track_dir / self._BASS_STEM
            if not bass_dir.is_dir():
                continue
            stem_files = sorted(
                p for p in bass_dir.iterdir() if p.suffix.lower() in self._AUDIO_EXTS
            )
            if not stem_files:
                continue
            tracks.append(
                {
                    "track_id": track_dir.name,
                    "stem_paths": stem_files,
                    "data_json": track_dir / "data.json"
                    if (track_dir / "data.json").is_file()
                    else None,
                }
            )
        return tracks

    # -- split assignment -------------------------------------------------

    @classmethod
    def _in_split(cls, track_id: str, split: str) -> bool:
        if split == "all":
            return True
        # Stable, platform-independent bucket in [0, 1) from track_id.
        digest = hashlib.sha1(track_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        if split == "train":
            return bucket < cls._TRAIN_FRAC
        if split == "valid":
            return cls._TRAIN_FRAC <= bucket < cls._VALID_FRAC
        return bucket >= cls._VALID_FRAC  # test

    # -- stem loading + normalization ------------------------------------

    def _load_and_mix(self, paths: list[Path]) -> torch.Tensor:
        """Load every stem file, resample, mono-collapse, sum. Returns (1, T_src)."""
        mixed: torch.Tensor | None = None
        for p in paths:
            wav, sr = torchaudio.load(str(p))
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.to(torch.float32, copy=False)

            if mixed is None:
                mixed = wav
            else:
                if wav.size(-1) > mixed.size(-1):
                    pad = wav.size(-1) - mixed.size(-1)
                    mixed = torch.nn.functional.pad(mixed, (0, pad))
                elif wav.size(-1) < mixed.size(-1):
                    pad = mixed.size(-1) - wav.size(-1)
                    wav = torch.nn.functional.pad(wav, (0, pad))
                mixed = mixed + wav
        assert mixed is not None  # _discover_tracks guarantees at least one file
        return mixed

    def _normalize_loudness(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply integrated-loudness normalization to `target_lufs` (EBU R128)."""
        if self.target_lufs is None:
            return wav
        meter = pyln.Meter(self.sample_rate)
        # pyloudnorm expects shape (samples,) or (samples, channels) — convert,
        # normalize, convert back. `.detach().cpu()` makes this function safe
        # to call on GPU / autograd tensors too (caller is CPU-only today, but
        # that's not an invariant the function should depend on). Peak-limit
        # at -1 dBFS to avoid clipping when scaling up quiet content.
        x = wav.detach().cpu().squeeze(0).numpy()
        try:
            loudness = meter.integrated_loudness(x)
        except ValueError as e:
            # pyloudnorm raises ValueError when the signal is shorter than its
            # 400 ms integration window. Narrow the catch to that specific
            # message so unrelated ValueErrors surface instead of producing a
            # silently-unnormalized track.
            if "signal" in str(e).lower() or "block" in str(e).lower():
                return wav
            raise
        if not np.isfinite(loudness):
            return wav
        normalized = pyln.normalize.loudness(x, loudness, self.target_lufs)
        peak = float(np.max(np.abs(normalized)))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
        return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)

    def _prepare_track(self, track: dict) -> dict:
        waveform = self._load_and_mix(track["stem_paths"])
        waveform = self._normalize_loudness(waveform)
        return {
            "track_id": track["track_id"],
            "waveform": waveform,  # (1, T_src)
            "length": waveform.size(-1),
        }

    # -- segment indexing -------------------------------------------------

    def _build_index(self) -> list[tuple[int, int, int]]:
        index: list[tuple[int, int, int]] = []
        for cache_idx, entry in enumerate(self._cache):
            length = entry["length"]
            if self.deterministic:
                # Hop through the stem at hop_samples until we pass the end.
                # Final partial segment is emitted and zero-padded at fetch time.
                seg = 0
                start = 0
                # Emit at least one segment per track even if shorter than the crop.
                while True:
                    index.append((cache_idx, start, seg))
                    seg += 1
                    start += self.hop_samples
                    if start + self.num_samples > length:
                        break
            else:
                # One virtual slot per track; the actual crop is re-drawn each
                # __getitem__ call using a seeded RNG derived from (seed, idx).
                index.append((cache_idx, 0, 0))
        return index

    # -- dataset API ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def _crop(self, waveform: torch.Tensor, start: int) -> torch.Tensor:
        """Extract `num_samples` from `waveform` starting at `start`; zero-pad the tail."""
        end = start + self.num_samples
        length = waveform.size(-1)
        if end <= length:
            return waveform[..., start:end].clone()
        segment = waveform[..., start:length]
        return torch.nn.functional.pad(segment, (0, self.num_samples - segment.size(-1)))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, MoisesDBLabel]:
        cache_idx, start, seg_idx = self._index[idx]
        entry = self._cache[cache_idx]
        waveform: torch.Tensor = entry["waveform"]
        length: int = entry["length"]

        if self.deterministic:
            crop_start = start
        else:
            # Per-item seeded RNG gives reproducibility across `DataLoader`
            # workers without sharing global state.
            rng = np.random.default_rng((self.seed, cache_idx, idx))
            max_start = max(0, length - self.num_samples)
            crop_start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

        wav = self._crop(waveform, crop_start)
        label: MoisesDBLabel = {
            "track_id": entry["track_id"],
            "track_idx": cache_idx,
            "stem": self._BASS_STEM,
            "segment_idx": seg_idx,
        }
        return wav, label
