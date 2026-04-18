import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class MoisesDB(Dataset):
    def __init__(self, data_dir, sr=16000, n_mels=128, duration=4.0, pitch_shift_range=(-6, 6)):
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.samples = int(sr * duration)
        self.pitch_shift_range = pitch_shift_range

        # path: data_dir/moisesdb_v0.1/<track_uuid>/bass/<file>.wav
        v01_dir = os.path.join(data_dir, "moisesdb_v0.1")

        self.files = []
        for track_id, track in enumerate(sorted(os.listdir(v01_dir))):
            bass_dir = os.path.join(v01_dir, track, "bass")
            if not os.path.isdir(bass_dir):
                continue
            for fname in os.listdir(bass_dir):
                if fname.endswith(".wav"):
                    self.files.append({
                        "path": os.path.join(bass_dir, fname),
                        "identity": track_id,
                        "track_name": track,
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]

        y, _ = librosa.load(item["path"], sr=self.sr, mono=True)
        y = self._random_crop(y)

        low, high = self.pitch_shift_range
        n_steps = np.random.randint(low, high + 1)
        y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

        mel = self._to_mel(y)
        mel_shifted = self._to_mel(y_shifted)

        return {
            "mel": torch.tensor(mel, dtype=torch.float32),
            "mel_shifted": torch.tensor(mel_shifted, dtype=torch.float32),
            "identity": torch.tensor(item["identity"], dtype=torch.long),
            "shift": torch.tensor(n_steps, dtype=torch.long),
        }

    def _to_mel(self, y):
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        return librosa.power_to_db(mel)

    def _random_crop(self, y):
        if len(y) <= self.samples:
            return np.pad(y, (0, self.samples - len(y)))
        start = np.random.randint(0, len(y) - self.samples)
        return y[start:start + self.samples]
