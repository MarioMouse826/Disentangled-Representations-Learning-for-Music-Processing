import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import argparse

from src.models.vae import SymmetryVAE
from src.models.beta_vae import BetaVAE
from src.models.hierarchical_vae import HierarchicalVAE
from src.data.zeroshot_dataset import ZeroShotDataset
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       type=str, required=True)
    p.add_argument("--preprocessed_dir", type=str, required=True)
    p.add_argument("--device",           type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]

    if "lambda_sym" in saved_args:
        model = SymmetryVAE(
            zs_dim=saved_args["zs_dim"],
            zc_dim=saved_args["zc_dim"],
            beta=saved_args["beta"],
            lambda_sym=saved_args["lambda_sym"]
        ).to(device)
    elif "zg_dim" in saved_args:
        model = HierarchicalVAE(
            zg_dim=saved_args["zg_dim"],
            zl_dim=saved_args["zl_dim"],
            beta=saved_args["beta"]
        ).to(device)
    else:
        model = BetaVAE(
            zs_dim=saved_args["zs_dim"],
            zc_dim=saved_args["zc_dim"],
            beta=saved_args["beta"]
        ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def run_zeroshot(model, loader, device):
    song_zs = defaultdict(list)
    song_recon_err = defaultdict(list)

    for batch in loader:
        mel = batch["mel"].to(device)
        songs = batch["song"]

        zs, zc = model.encode(mel)
        mel_recon = model.decode(zs, zc)

        recon_err = F.mse_loss(mel_recon, mel, reduction='none').mean(dim=[1, 2]).cpu().numpy()

        for i, song in enumerate(songs):
            song_zs[song].append(zs[i].cpu().numpy())
            song_recon_err[song].append(recon_err[i])

    print("\n=== Zero-Shot Evaluation Results ===")
    print(f"{'Song':<35} {'Chunks':>6} {'Recon Err':>10} {'Style Var':>10}")
    print("-" * 65)

    all_vars = []
    all_errs = []

    for song in sorted(song_zs.keys()):
        zs_stack = np.stack(song_zs[song])
        style_var = np.mean(np.var(zs_stack, axis=0))
        recon_err = np.mean(song_recon_err[song])

        all_vars.append(style_var)
        all_errs.append(recon_err)

        print(f"{song:<35} {len(zs_stack):>6} {recon_err:>10.4f} {style_var:>10.4f}")

    print("-" * 65)
    print(f"{'MEAN':<35} {'':>6} {np.mean(all_errs):>10.4f} {np.mean(all_vars):>10.4f}")
    print("\nStyle Var = variance of zs across chunks of same song (lower = more stable = better)")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.checkpoint, device)

    dataset = ZeroShotDataset(args.preprocessed_dir)
    print(f"Zero-shot dataset: {len(dataset)} chunks across 8 songs")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    run_zeroshot(model, loader, device)
