import os
import argparse
import torch
import torch.optim as optim

from src.models.vae import SymmetryVAE
from src.data.dataloader import get_train_loader
from src.data.nsynth_preprocessed import NSynthBassPreprocessed
from src.data.moisesdb_preprocessed import MoisesDBPreprocessed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nsynth_dir",   type=str, default="data/nsynth")
    p.add_argument("--moisesdb_dir", type=str, default="data/moisesdb")
    p.add_argument("--zs_dim",       type=int, default=32)
    p.add_argument("--zc_dim",       type=int, default=32)
    p.add_argument("--beta",         type=float, default=1.0)
    p.add_argument("--lambda_sym",   type=float, default=1.0)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",       type=str, default=None)
    return p.parse_args()


def load_datasets(args):
    nsynth_ds, moisesdb_ds = None, None

    if os.path.exists(args.nsynth_dir):
        nsynth_ds = NSynthBassPreprocessed(args.nsynth_dir)
        print(f"NSynth: {len(nsynth_ds)} samples")
    else:
        print(f"NSynth dir not found ({args.nsynth_dir}), skipping")

    if os.path.exists(args.moisesdb_dir):
        moisesdb_ds = MoisesDBPreprocessed(args.moisesdb_dir)
        print(f"MoisesDB: {len(moisesdb_ds)} samples")
    else:
        print(f"MoisesDB dir not found ({args.moisesdb_dir}), skipping")

    if nsynth_ds is None and moisesdb_ds is None:
        raise RuntimeError("No datasets found.")

    return nsynth_ds, moisesdb_ds


def train(args):
    device = torch.device(args.device)
    print(f"Device: {device}")

    nsynth_ds, moisesdb_ds = load_datasets(args)
    loader = get_train_loader(nsynth_ds, moisesdb_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = SymmetryVAE(
        zs_dim=args.zs_dim,
        zc_dim=args.zc_dim,
        beta=args.beta,
        lambda_sym=args.lambda_sym
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    os.makedirs("results/checkpoints", exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        skipped = 0

        for i, batch in enumerate(loader):
            mel         = batch["mel"].to(device)
            mel_shifted = batch["mel_shifted"].to(device)

            optimizer.zero_grad()
            _, loss, components = model(mel, mel_shifted)
            loss.backward()

            # skip batch if gradients are NaN
            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                print(f"NaN gradient at epoch {epoch} batch {i+1}, skipping")
                optimizer.zero_grad()
                skipped += 1
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % args.log_every == 0:
                print(
                    f"Epoch {epoch} | Batch {i+1}/{len(loader)} | "
                    f"total={components['total']:.4f} "
                    f"recon={components['lrecon']:.4f} "
                    f"kl_zs={components['kl_zs']:.4f} "
                    f"kl_zc={components['kl_zc']:.4f} "
                    f"sym={components['lsym']:.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} complete | avg loss: {avg_loss:.4f} | skipped batches: {skipped}")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            }, f"results/checkpoints/epoch_{epoch}.pt")


if __name__ == "__main__":
    args = parse_args()
    train(args)
