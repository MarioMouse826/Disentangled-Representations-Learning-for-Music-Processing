"""Hydra entrypoint for SC-VAE + baseline training (plan Task 4.2).

Composes `model`, `data`, `lit`, and `trainer` from YAML and runs
`trainer.fit`. Intended usage:

    python -m src.training.cli                                # default SC-VAE on synthetic
    python -m src.training.cli model=beta_vae                 # swap baseline
    python -m src.training.cli data=nsynth_bass               # real data
    python -m src.training.cli +trainer.fast_dev_run=true     # 1-batch smoke
    python -m src.training.cli lit.beta_s=4.0 lit.lambda_swap=0.5

Config tree lives in `/configs/train/base.yaml` + per-module overrides in
`/configs/model/*.yaml` and `/configs/data/*.yaml`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.utils.compute_budget import BudgetTracker
from src.utils.manifest import write_manifest
from src.utils.seed import set_seeds


_CONFIG_DIR = str((Path(__file__).resolve().parents[2] / "configs").as_posix())


def _collate(batch: list[dict]) -> dict:
    """Stack equal-shape tensor dict items. Recurses into nested dicts
    (e.g. labels) and falls back to `torch.as_tensor` for scalars."""
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], dict):
            out[k] = _collate(vals)
        else:
            out[k] = torch.as_tensor(vals)
    return out


def _build_loader(dataset: Any, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=_collate,
        # drop_last=True: β-TCVAE requires B>=2 per batch; drop tail for
        # uniform step count across all models.
        drop_last=True,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def run(cfg: DictConfig) -> float:
    """Library-style entrypoint — usable from tests without going through CLI."""
    seed = int(cfg.get("seed", 0))
    set_seeds(seed)
    pl.seed_everything(seed, workers=True)

    model = hydra.utils.instantiate(cfg.model)
    train_ds = hydra.utils.instantiate(cfg.data.train_dataset)
    val_ds = hydra.utils.instantiate(cfg.data.val_dataset)

    train_loader = _build_loader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        shuffle=True,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        shuffle=False,
    )

    lit = hydra.utils.instantiate(cfg.lit, model=model)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = pl.Trainer(**trainer_kwargs)

    with BudgetTracker() as tracker:
        trainer.fit(lit, train_loader, val_loader)

    run_dir = Path(trainer.log_dir or ".")
    write_manifest(
        config=OmegaConf.to_container(cfg, resolve=True),
        seed=seed,
        out_path=run_dir / "run_manifest.json",
        budget=tracker.summary(),
    )

    # Return final train/loss for caller convenience (e.g. sweep early-stop).
    metrics = trainer.callback_metrics
    return float(metrics.get("train/loss", torch.tensor(float("nan"))))


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="base")
def main(cfg: DictConfig) -> float:
    return run(cfg)


if __name__ == "__main__":
    main()
