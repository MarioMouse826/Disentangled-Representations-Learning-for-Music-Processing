import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class NSynthWithIdentity(Dataset):
    """Wraps NSynthBass and adds a dummy identity=-1"""
    def __init__(self, nsynth_dataset):
        self.ds = nsynth_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample["identity"] =  tensor(-1, dtype=torch.long)
        return sample


class MoisesDBWithPitch(Dataset):
    """Wraps MoisesDB and adds a dummy pitch=-1"""
    def __init__(self, moisesdb_dataset):
        self.ds = moisesdb_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample["pitch"] = torch.tensor(-1, dtype=torch.long)
        return sample


def get_train_loader(nsynth_ds, moisesdb_ds, batch_size=32, num_workers=0):
    datasets = []
    if nsynth_ds is not None:
        datasets.append(NSynthWithIdentity(nsynth_ds))
    if moisesdb_ds is not None:
        datasets.append(MoisesDBWithPitch(moisesdb_ds))
    if not datasets:
        raise RuntimeError("No datasets provided")
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def get_eval_loader(dataset, batch_size=64, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)