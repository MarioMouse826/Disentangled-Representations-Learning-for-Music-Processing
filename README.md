# Symmetry-Constrained VAE for Audio Disentanglement

Implementation of a Symmetry-Constrained Variational Autoencoder (SC-VAE) that disentangles timbre (`z_s`) from pitch/content (`z_c`) in monophonic bass audio via pitch-shift *invariance* on the style subspace and pitch-shift *equivariance* on the content subspace. Group structure follows Higgins et al. (2018) "Towards a Definition of Disentangled Representations" (arXiv:1812.02230).


## Setup

```bash
# system deps (macOS arm64; skip on Linux)
brew install libsndfile rubberband

# virtual env (python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# install python deps
pip install -r requirements.txt

# macOS arm64 libsndfile shim (once): link brew's libsndfile into soundfile package
ln -sf /opt/homebrew/lib/libsndfile.dylib .venv/lib/python3.11/site-packages/_soundfile_data/libsndfile.dylib
```

## Quick start

```bash
# run full test suite
make test

# lint
make lint

# train SC-VAE on NSynth bass
make train-scvae

# run all evaluation metrics (MIG, DCI, SAP, Modularity, SRR, IR, ER, PitchAcc@50)
make eval-all
```

## Project structure

```
src/
  data/          # NSynth, MoisesDB, zero-shot loaders + pitch-shift group action
  models/        # encoder, style/content heads, rho(g), decoder, baselines, sc_vae
  losses/        # reconstruction, KL, TC, symmetry (inv/equi/swap)
  training/      # PyTorch Lightning module, hydra CLI, callbacks
  evaluation/    # MIG, DCI, SAP, Modularity, FactorVAE-score, SRR, IR/ER, CREPE pitch acc
  utils/
tests/           # unit + integration (>=85% coverage)
configs/         # hydra configs (data/model/train/eval)
```

## Baselines

- β-VAE (Higgins et al. 2017)
- β-TCVAE (Chen et al. 2018, arXiv:1803.05428)
- FactorVAE (Kim & Mnih 2018, arXiv:1802.04942)
- Autoregressive Hierarchical VAE (Ladder VAE / NVAE-style)

## Datasets

- NSynth bass subset (Engel et al. 2017, CC-BY-4.0)
- MoisesDB bass stems (2024)
- Zero-shot OOD: curated famous bass lines

## License

TBD.
