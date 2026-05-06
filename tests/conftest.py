"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# Minimal NSynth manifest schema: see https://magenta.tensorflow.org/datasets/nsynth
# We synthesize 3 notes (2 bass, 1 non-bass) across 2 splits to exercise filtering + split switching.
_SR = 16000
_DUR = 4.0


def _write_wav(path: Path, pitch_midi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic sinusoid at the MIDI pitch frequency — gives the fixture audible content
    # that downstream pitch-shift tests can rely on without being purely silent.
    freq = 440.0 * 2.0 ** ((pitch_midi - 69) / 12.0)
    t = np.arange(int(_SR * _DUR)) / _SR
    wave = (0.25 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, _SR, subtype="PCM_16")


def _manifest_entry(key: str, pitch: int, family: str, source: str, inst_id: int) -> dict:
    return {
        "note_str": key,
        "pitch": pitch,
        "velocity": 100,
        "instrument_family_str": family,
        "instrument_source_str": source,
        "instrument": inst_id,
        "sample_rate": _SR,
        "qualities_str": [],
    }


@pytest.fixture
def nsynth_fixture(tmp_path: Path) -> Path:
    """Creates a minimal NSynth-layout directory tree under tmp_path.

    Returns root that can be passed to NSynthBass(root=...).
    Tree:
        <root>/nsynth-train/examples.json
        <root>/nsynth-train/audio/<key>.wav  (3 notes: 2 bass + 1 guitar)
        <root>/nsynth-valid/examples.json    (1 bass note)
        <root>/nsynth-valid/audio/<key>.wav
    """
    for split, entries in {
        "train": [
            ("bass_acoustic_000-040-100", 40, "bass", "acoustic", 0),
            ("bass_electronic_001-055-100", 55, "bass", "electronic", 1),
            ("guitar_acoustic_002-060-100", 60, "guitar", "acoustic", 2),
        ],
        "valid": [
            ("bass_synthetic_003-030-100", 30, "bass", "synthetic", 3),
        ],
    }.items():
        split_dir = tmp_path / f"nsynth-{split}"
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, dict] = {}
        for key, pitch, family, source, inst_id in entries:
            _write_wav(audio_dir / f"{key}.wav", pitch)
            manifest[key] = _manifest_entry(key, pitch, family, source, inst_id)
        with open(split_dir / "examples.json", "w") as f:
            json.dump(manifest, f)
    return tmp_path


def _write_bass_stem(path: Path, duration_s: float, freq: float, sr: int = 44100) -> None:
    """Write a sinusoid at `freq` Hz lasting `duration_s` seconds at `sr`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * duration_s)) / sr
    wave = (0.3 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sr, subtype="PCM_16")


def _write_harmonic_bass(
    path: Path,
    duration_s: float,
    freq: float,
    sr: int = 16000,
    n_partials: int = 5,
) -> None:
    """Write a harmonic stack at `freq` with amplitude decay ~1/k.

    CREPE is trained on music audio and relies on harmonic structure; a pure
    sinusoid gives a flat softmax over pitch bins and returns -inf periodicity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * duration_s)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for k in range(1, n_partials + 1):
        wave += (0.4 / k) * np.sin(2.0 * np.pi * k * freq * t).astype(np.float32)
    # Normalize peak to avoid clipping.
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave = wave / peak * 0.8
    sf.write(str(path), wave.astype(np.float32), sr, subtype="PCM_16")


@pytest.fixture
def moisesdb_fixture(tmp_path: Path) -> Path:
    """Creates a minimal MoisesDB-like tree under tmp_path.

    Layout mirrors the public release — each track is a UUID folder containing
    `data.json` plus stem subfolders; `bass/` holds one or more wav files that
    must be summed to produce the bass signal.

    Tree:
        <root>/track-A/data.json
        <root>/track-A/bass/di.wav          (10 s @ 44.1 kHz, 110 Hz sine)
        <root>/track-A/bass/amped.wav       (10 s @ 44.1 kHz, 110 Hz sine, slightly louder)
        <root>/track-A/drums/kick.wav       (unused — must be ignored by loader)
        <root>/track-B/data.json
        <root>/track-B/bass/di.wav          (6 s @ 44.1 kHz, 82 Hz sine)
        <root>/track-C/data.json            (no bass folder — must be skipped)
        <root>/track-C/vocals/lead.wav
    """
    import json as _json

    tracks = {
        "track-A": [("bass/di.wav", 10.0, 110.0), ("bass/amped.wav", 10.0, 110.0)],
        "track-B": [("bass/di.wav", 6.0, 82.0)],
        "track-C": [("vocals/lead.wav", 4.0, 440.0)],
    }
    for tid, stems in tracks.items():
        tdir = tmp_path / tid
        tdir.mkdir()
        with open(tdir / "data.json", "w") as f:
            _json.dump({"id": tid, "artist": "fixture"}, f)
        # Drums file for track-A so we can test "non-bass is ignored".
        if tid == "track-A":
            _write_bass_stem(tdir / "drums" / "kick.wav", 2.0, 60.0)
        for rel, dur, freq in stems:
            _write_bass_stem(tdir / rel, dur, freq)
    return tmp_path


@pytest.fixture
def zero_shot_input_fixture(tmp_path: Path) -> Path:
    """3 toy bass-line wavs at 16 kHz, monophonic sinusoids of varying pitch/length.

    Duration choices are tuned so the preparer emits multiple segments per clip at
    the default 4 s / 2 s hop setting — exercises windowing logic.
    """
    # Frequencies pinned above 80 Hz — CREPE-tiny on pure sines is unreliable
    # below that range; real bass has harmonics that make low pitches detectable,
    # but the fixture is pure sinusoid.
    wavs = [
        ("money.wav", 10.0, 110.0),          # A2 → 4 segments @ 4s/2s
        ("under_pressure.wav", 6.0, 98.0),   # G2 → 2 segments
        ("stand_by_me.wav", 4.0, 82.41),     # E2 → 1 segment exact-fit
    ]
    input_dir = tmp_path / "input_stems"
    input_dir.mkdir()
    for name, dur, freq in wavs:
        _write_harmonic_bass(input_dir / name, dur, freq, sr=16000, n_partials=5)
    return input_dir


@pytest.fixture
def zero_shot_shard_dir(tmp_path: Path, zero_shot_input_fixture: Path) -> Path:
    """Runs the prepare script on the input fixture and returns the shard directory."""
    from src.data.zero_shot_prepare import prepare_zero_shot_shards

    shard_dir = tmp_path / "shards"
    prepare_zero_shot_shards(
        input_dir=zero_shot_input_fixture,
        output_dir=shard_dir,
        sample_rate=16000,
        segment_seconds=4.0,
        hop_seconds=2.0,
        crepe_step_ms=10.0,
        device="cpu",
    )
    return shard_dir

