import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NSynthBassPreprocessed(Dataset):
    def __init__(self, data_dir, pitch_shift_range=(-6, 6)):
        self.data_dir = data_dir
        self.pitch_shift_range = pitch_shift_range

        self.files = []
        for fname in os.listdir(data_dir):
            if not fname.endswith(".npy"):
                continue
            if not fname.startswith("bass"):
                continue
            try:
                pitch = int(fname.split("-")[1])
            except (IndexError, ValueError):
                continue
            self.files.append({"path": os.path.join(data_dir, fname), "pitch": pitch})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        mel = np.load(item["path"])

        low, high = self.pitch_shift_range
        n_steps = np.random.randint(low, high + 1)
        mel_shifted = self._pitch_shift_mel(mel, n_steps)

        return {
            "mel":         torch.tensor(mel, dtype=torch.float32),
            "mel_shifted": torch.tensor(mel_shifted, dtype=torch.float32),
            "pitch":       torch.tensor(item["pitch"], dtype=torch.long),
            "shift":       torch.tensor(n_steps, dtype=torch.long),
        }

    def _pitch_shift_mel(self, mel, n_steps):
        if n_steps == 0:
            return mel.copy()
        return np.roll(mel, n_steps, axis=0).astype(np.float32)
