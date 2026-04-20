import os
import numpy as np
import torch
from torch.utils.data import Dataset

ZEROSHOT_IDS = [
    ("04f22a24", "blues",           "Iain Kerr & Friends", "Can't Play The Blues"),
    ("0e0d57cd", "bossa_nova",      "Firefly",             "The Best In Me"),
    ("3f5233cb", "country",         "Firefly",             "The Last To Know"),
    ("28748b6e", "electronic",      "Holy Magick",         "Stolen Car"),
    ("1e957e76", "jazz",            "Iain Kerr & Friends", "Dreaming Bout Being With You"),
    ("3a047d1a", "musical_theatre", "Firefly",             "Places"),
    ("b876b54b", "reggae",          "Centenary",           "Sick Of Waiting"),
    ("8f36f17f", "world_folk",      "Juniper",             "Nexus"),
]


class ZeroShotDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.samples = []
        self.metadata = []

        for uid, genre, artist, song in ZEROSHOT_IDS:
            # find the full folder name starting with uid
            matches = [
                d for d in os.listdir(preprocessed_dir)
                if d.startswith(uid)
            ]
            if not matches:
                print(f"Warning: no folder found for {uid}")
                continue
            folder = os.path.join(preprocessed_dir, matches[0])
            files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
            for fname in files:
                self.samples.append(os.path.join(folder, fname))
                self.metadata.append({
                    "uid": uid,
                    "genre": genre,
                    "artist": artist,
                    "song": song
                }) 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel = np.load(self.samples[idx])
        mel = torch.tensor(mel, dtype=torch.float32)
        meta = self.metadata[idx]
        return {
            "mel": mel,
            "uid": meta["uid"],
            "genre": meta["genre"],
            "song": meta["song"],
        }
