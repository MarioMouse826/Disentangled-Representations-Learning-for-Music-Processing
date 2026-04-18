import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MoisesDBPreprocessed(Dataset):
    def __init__(self, data_dir, pitch_shift_range=(-6, 6)):
        self.data_dir = data_dir
        self.pitch_shift_range = pitch_shift_range

        self.files = []
        for track_id, track in enumerate(sorted(os.listdir(data_dir))):
            track_dir = os.path.join(data_dir, track)
            if not os.path.isdir(track_dir):
                continue
            for fname in os.listdir(track_dir):
                if fname.endswith(".npy"):
                    self.files.append({
                        "path": os.path.join(track_dir, fname),
                        "identity": track_id,
                    })

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
            "identity":    torch.tensor(item["identity"], dtype=torch.long),
            "shift":       torch.tensor(n_steps, dtype=torch.long),
        }

    def _pitch_shift_mel(self, mel, n_steps):
        if n_steps == 0:
            return mel.copy()
        return np.roll(mel, n_steps, axis=0).astype(np.float32)
