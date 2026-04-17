import os
import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class NSynthBass(Dataset):
    def __init__(self, data_dir, sr=16000, n_mels=128, duration=4.0, pitch_shift_range=(-6, 6)):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.samples = int(sr * duration)
        self.pitch_shift_range = pitch_shift_range

        # load metadata and filter bass only
        meta_path = os.path.join(data_dir, "examples.json")
        with open(meta_path) as f:
            meta = json.load(f)

        self.files = [
            {"name": k, "pitch": v["pitch"]}
            for k, v in meta.items()
            if "bass" in v["instrument_family_str"]
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        path = os.path.join(self.data_dir, "audio", item["name"] + ".wav")

        y, _ = librosa.load(path, sr=self.sr, mono=True)
        y = self._fix_length(y)

        # random pitch shift for symmetry pair
        n_steps = np.random.randint(*self.pitch_shift_range)
        y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        mel = self._to_mel(y)
        mel_shifted = self._to_mel(y_shifted)

        return {
            "mel": torch.tensor(mel, dtype=torch.float32),
            "mel_shifted": torch.tensor(mel_shifted, dtype=torch.float32),
            "pitch": torch.tensor(item["pitch"], dtype=torch.long),
            "shift": torch.tensor(n_steps, dtype=torch.long),
        }

    def _to_mel(self, y):
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        return librosa.power_to_db(mel)

    def _fix_length(self, y):
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        return y