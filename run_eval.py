import torch
from src.models.vae import SymmetryVAE
from src.data.nsynth_preprocessed import NSynthBassPreprocessed
from src.data.moisesdb_preprocessed import MoisesDBPreprocessed
from torch.utils.data import DataLoader
from src.evaluation.evaluate import evaluate
import argparse

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=str, required=True)
p.add_argument("--nsynth_dir", type=str, required=True)
p.add_argument("--moisesdb_dir", type=str, required=True)
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

device = torch.device(args.device)

ckpt = torch.load(args.checkpoint, map_location=device)
saved_args = ckpt["args"]

from src.models.beta_vae import BetaVAE
from src.models.vae import SymmetryVAE

if "lambda_sym" in saved_args:
    model = SymmetryVAE(
        zs_dim=saved_args["zs_dim"],
        zc_dim=saved_args["zc_dim"],
        beta=saved_args["beta"],
        lambda_sym=saved_args["lambda_sym"]
    ).to(device)
else:
    model = BetaVAE(
        zs_dim=saved_args["zs_dim"],
        zc_dim=saved_args["zc_dim"],
        beta=saved_args["beta"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

nsynth_ds   = NSynthBassPreprocessed(args.nsynth_dir)
moisesdb_ds = MoisesDBPreprocessed(args.moisesdb_dir)

nsynth_loader   = DataLoader(nsynth_ds,   batch_size=256, shuffle=False, num_workers=4)
moisesdb_loader = DataLoader(moisesdb_ds, batch_size=256, shuffle=False, num_workers=4)

evaluate(model, nsynth_loader, moisesdb_loader, device=str(device))
