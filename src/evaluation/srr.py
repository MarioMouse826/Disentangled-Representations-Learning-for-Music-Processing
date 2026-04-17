import numpy as np
import torch


def srr(mel_orig, mel_recon, max_db=60.0):
    """
    Signal-to-Reconstruction Ratio in dB.

    Args:
        mel_orig:  np.ndarray or torch.Tensor of shape (N, n_mels, T) or (n_mels, T)
        mel_recon: same shape as mel_orig
        max_db:    clip value for perfect reconstruction (default 60dB)

    Returns:
        float — mean SRR in dB across batch
    """
    if isinstance(mel_orig, torch.Tensor):
        mel_orig = mel_orig.detach().cpu().numpy()
    if isinstance(mel_recon, torch.Tensor):
        mel_recon = mel_recon.detach().cpu().numpy()

    # flatten to (N, -1) if batched, or (1, -1) if single sample
    if mel_orig.ndim == 2:
        mel_orig = mel_orig[np.newaxis]
        mel_recon = mel_recon[np.newaxis]

    mel_orig = mel_orig.reshape(len(mel_orig), -1)
    mel_recon = mel_recon.reshape(len(mel_recon), -1)

    signal_power = np.sum(mel_orig ** 2, axis=1)
    noise_power = np.sum((mel_orig - mel_recon) ** 2, axis=1)

    srr_per_sample = 10 * np.log10(signal_power / (noise_power + 1e-10))
    srr_per_sample = np.clip(srr_per_sample, a_min=None, a_max=max_db)

    return float(np.mean(srr_per_sample))