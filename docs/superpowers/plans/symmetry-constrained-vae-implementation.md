# Symmetry-Constrained VAE for Audio Disentanglement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and evaluate a Symmetry-Constrained Variational Autoencoder (SC-VAE) that disentangles timbre (`z_s`) from pitch/content (`z_c`) in monophonic bass audio by enforcing pitch-shift *invariance* on the style subspace and pitch-shift *equivariance* on the content subspace, surpassing β-VAE / FactorVAE / AR-Hierarchical-VAE baselines on MIG, DCI-Disentanglement, and Signal-to-Reconstruction Ratio, with a publication-grade empirical protocol targeting NeurIPS / ICML / ICLR.

**Architecture:** Dual-encoder VAE over log-mel spectrograms (128 mels, 16 kHz, 25 ms / 10 ms hop). Style encoder outputs a *time-pooled* posterior `q(z_s|x)` enforced to be invariant to the group action `T_g` (pitch shift by `g` cents). Content encoder outputs a *time-resolved* posterior `q(z_c|x)` whose representation is enforced to be equivariant: `z_c(T_g x) = ρ(g) z_c(x)` via a linear group-representation head `ρ(g)`. Decoder is a HiFi-GAN-style convolutional vocoder-free decoder predicting the mel-spectrogram, with a separately trained HiFi-GAN / BigVGAN head for waveform reconstruction. Group structure follows Higgins et al. (2018) "Towards a Definition of Disentangled Representations" (arXiv:1812.02230). Evaluation follows Eastwood & Williams (DCI, 2018) and Chen et al. (β-TCVAE / MIG, 2018, arXiv:1802.04942) with the Locatello (2019, arXiv:1811.12359) impossibility caveat addressed by using ground-truth MIDI pitch and track identity as supervised factors for assessment only.

**Tech Stack:** PyTorch 2.3+, torchaudio, PyTorch Lightning 2.x, librosa, pyrubberband (high-quality pitch shift with formant preservation), hydra-core (config), Weights & Biases (experiment tracking), scikit-learn (DCI classifiers, MI estimators), NSynth (Engel et al. 2017, arXiv:1704.01279) for the bass subset, MoisesDB (2024) for real-world multi-track stems, HiFi-GAN / BigVGAN pretrained checkpoint for waveform rendering, DDSP (Engel et al. 2020, arXiv:2001.04643) as an alternative differentiable synthesizer decoder for an ablation arm.

**Target venue claims:** (a) application of formal group-equivariant VAE framework (Higgins et al. 2018, arXiv:1812.02230) to pitch/timbre disentanglement in audio, using block-diagonal SO(2) rotational representation `ρ(g)` on `z_c` — contrasts with DISMIX (Luo et al. 2024, arXiv:2408.10807) which tackles mixture separation without group-theoretic formalism, and with GES-VAE which uses non-rotational group structure; (b) new zero-shot OOD benchmark built from famous bass lines; (c) ablation showing symmetry prior beats β-penalty at matched reconstruction quality; (d) open-source release of code, configs, checkpoints, and assessment harness; (e) novel swap-consistency loss `L_swap` constraining decoder on cross-factor recomposition; (f) explicit inductive-bias declaration circumventing Locatello impossibility (arXiv:1811.12359) — supervision comes from known pitch-shift group structure, not factor labels.

---

## File Structure

```
ml_project/
├── configs/
│   ├── data/
│   │   ├── nsynth_bass.yaml
│   │   ├── moisesdb_bass.yaml
│   │   └── zero_shot_famous.yaml
│   ├── model/
│   │   ├── beta_vae.yaml
│   │   ├── factor_vae.yaml
│   │   ├── ar_hvae.yaml
│   │   └── sc_vae.yaml
│   ├── train/
│   │   ├── base.yaml
│   │   └── sweep.yaml
│   └── eval/
│       ├── mig.yaml
│       ├── dci.yaml
│       └── srr.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nsynth.py           # NSynth bass loader + MIDI-pitch label extraction
│   │   ├── moisesdb.py         # MoisesDB stem loader + track-ID label
│   │   ├── zero_shot.py        # famous-bass-line loader (OOD probes)
│   │   ├── augment.py          # pitch-shift group action T_g (rubberband-backed)
│   │   └── features.py         # log-mel frontend (STFT, mel filterbank)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py          # shared conv trunk
│   │   ├── style_head.py       # invariance head -> q(z_s|x)
│   │   ├── content_head.py     # equivariance head -> q(z_c|x)
│   │   ├── group_repr.py       # learnable rho(g) on z_c
│   │   ├── decoder.py          # mel-spectrogram decoder
│   │   ├── vocoder.py          # HiFi-GAN wrapper (inference only, pretrained)
│   │   ├── ddsp_decoder.py     # optional ablation decoder
│   │   ├── baselines/
│   │   │   ├── beta_vae.py
│   │   │   ├── factor_vae.py   # incl. discriminator for TC term
│   │   │   └── ar_hvae.py
│   │   └── sc_vae.py           # full Symmetry-Constrained VAE
│   ├── losses/
│   │   ├── recon.py            # multi-scale spectral + L1-mel
│   │   ├── kl.py               # Gaussian KL, free-bits, warmup, cyclical
│   │   ├── tc.py               # total correlation (FactorVAE + beta-TCVAE)
│   │   └── symmetry.py         # L_sym_inv, L_sym_equiv, commutative-diagram loss
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lit_module.py       # PyTorch Lightning wrapper
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   ├── callbacks.py        # KL annealing, EMA, audio logging
│   │   └── cli.py              # hydra entrypoint `python -m src.training.cli`
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── mig.py              # Mutual Information Gap
│   │   ├── dci.py              # Disentanglement / Completeness / Informativeness
│   │   ├── srr.py              # Signal-to-Reconstruction Ratio (log-spectral)
│   │   ├── equivariance_err.py # norm(rho(g)z_c(x) - z_c(T_g x)) and invariance gap
│   │   ├── latent_traversal.py # interpolation grids + audio exports
│   │   └── report.py           # aggregates all metrics -> LaTeX table
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── logging_utils.py
├── tests/
│   ├── data/
│   │   ├── test_nsynth_loader.py
│   │   ├── test_augment_group_action.py
│   │   └── test_features.py
│   ├── models/
│   │   ├── test_encoder_shapes.py
│   │   ├── test_group_repr.py
│   │   ├── test_sc_vae_forward.py
│   │   └── test_baselines.py
│   ├── losses/
│   │   ├── test_kl.py
│   │   ├── test_tc.py
│   │   └── test_symmetry.py
│   └── evaluation/
│       ├── test_mig.py
│       ├── test_dci.py
│       └── test_srr.py
├── scripts/
│   ├── download_nsynth.sh
│   ├── download_moisesdb.py
│   ├── prepare_zero_shot.py
│   ├── train_all_baselines.sh
│   └── run_full_eval.sh
├── docs/
│   ├── superpowers/plans/symmetry-constrained-vae-implementation.md   # this file
│   ├── paper/                  # LaTeX sources
│   └── figures/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

**Responsibility rules**
- `src/data/*` returns `(waveform, label_dict)` tuples only; never model-specific tensors.
- `src/models/*` never depends on `src/training/*`.
- Every loss returns a *scalar* and a *dict of unreduced components* for logging.
- Every assessment metric module exposes `compute(model, dataloader, device) -> dict`.
- All config defaults live in `configs/`; no magic numbers in code.

---

## Phase 0 — Repository Bootstrap

### Task 0.1: Initialize repository, Python, and tooling

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `Makefile`, `.gitignore`, `README.md`, `.pre-commit-config.yaml`

- [ ] **Step 1: Write `pyproject.toml` with pinned versions**

```toml
[project]
name = "sc-vae-audio"
version = "0.1.0"
description = "Symmetry-Constrained VAE for audio disentanglement"
requires-python = ">=3.10,<3.12"

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E","F","I","B","UP","NPY","RUF"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers"
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch==2.3.1
torchaudio==2.3.1
pytorch-lightning==2.3.3
hydra-core==1.3.2
omegaconf==2.3.0
librosa==0.10.2
soundfile==0.12.1
pyrubberband==0.3.0
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.1
pandas==2.2.2
wandb==0.17.5
einops==0.8.0
pretty_midi==0.2.10
tqdm==4.66.4
pytest==8.2.2
pytest-cov==5.0.0
ruff==0.5.5
```

- [ ] **Step 3: Write `Makefile`**

```make
.PHONY: install test lint fmt train eval
install:
	pip install -r requirements.txt
	pre-commit install
test:
	pytest --cov=src --cov-report=term-missing
lint:
	ruff check src tests
fmt:
	ruff format src tests
train-scvae:
	python -m src.training.cli model=sc_vae data=nsynth_bass
eval-all:
	bash scripts/run_full_eval.sh
```

- [ ] **Step 4: Commit**

```bash
git init
git add .
git commit -m "chore: bootstrap python project and tooling"
```

### Task 0.2: Create package skeleton with `__init__.py` files

- [ ] **Step 1: Create empty `__init__.py` in every package directory listed in the file structure**
- [ ] **Step 2: Add a smoke test `tests/test_import.py`**

```python
def test_imports():
    import src
    import src.data
    import src.models
    import src.losses
    import src.training
    import src.evaluation
```

- [ ] **Step 3: Run `pytest tests/test_import.py -v` — expect PASS**
- [ ] **Step 4: Commit**

```bash
git add src/ tests/test_import.py
git commit -m "chore: package skeleton"
```

---

## Phase 1 — Data Pipeline

### Task 1.1: NSynth bass subset loader

**Reference:** Engel et al. 2017, *Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders* (arXiv:1704.01279). Bass instrument family `bass`; ~68k notes, 4-second clips, 16 kHz mono, MIDI pitch 21–108.

**Files:**
- Create: `src/data/nsynth.py`
- Test: `tests/data/test_nsynth_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_nsynth_loader.py
import torch
from src.data.nsynth import NSynthBass

def test_nsynth_bass_item_schema(tmp_path, nsynth_fixture):
    ds = NSynthBass(root=nsynth_fixture, split="train", sample_rate=16000, duration=4.0)
    assert len(ds) > 0
    wav, label = ds[0]
    assert isinstance(wav, torch.Tensor)
    assert wav.shape == (1, 16000 * 4)
    assert label["pitch"].dtype == torch.long
    assert 21 <= int(label["pitch"]) <= 108
    assert "velocity" in label
    assert "instrument_source" in label
```

- [ ] **Step 2: Add `nsynth_fixture` in `tests/conftest.py` synthesizing 3 dummy WAVs and a minimal JSON manifest mimicking NSynth layout**
- [ ] **Step 3: Run `pytest tests/data/test_nsynth_loader.py -v` — expect FAIL (module missing)**
- [ ] **Step 4: Implement**

```python
# src/data/nsynth.py
import json
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset

class NSynthBass(Dataset):
    SOURCES = {"acoustic": 0, "electronic": 1, "synthetic": 2}

    def __init__(self, root, split="train", sample_rate=16000, duration=4.0):
        self.root = Path(root) / f"nsynth-{split}"
        self.sr = sample_rate
        self.n = int(sample_rate * duration)
        with open(self.root / "examples.json") as f:
            meta = json.load(f)
        self.items = [
            (k, v) for k, v in meta.items()
            if v["instrument_family_str"] == "bass"
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        k, v = self.items[idx]
        wav, sr = torchaudio.load(self.root / "audio" / f"{k}.wav")
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav[:1, : self.n]
        if wav.shape[-1] < self.n:
            wav = torch.nn.functional.pad(wav, (0, self.n - wav.shape[-1]))
        label = {
            "pitch": torch.tensor(v["pitch"], dtype=torch.long),
            "velocity": torch.tensor(v["velocity"], dtype=torch.long),
            "instrument_source": torch.tensor(
                self.SOURCES[v["instrument_source_str"]], dtype=torch.long
            ),
            "instrument_id": torch.tensor(v["instrument"], dtype=torch.long),
        }
        return wav, label
```

- [ ] **Step 5: Re-run test — expect PASS**
- [ ] **Step 6: Commit**

```bash
git add src/data/nsynth.py tests/data/test_nsynth_loader.py tests/conftest.py
git commit -m "feat(data): NSynth bass subset loader"
```

### Task 1.2: MoisesDB bass-stem loader

**Reference:** MoisesDB 2024 (Pereira et al.) — 240 multi-track masters with labeled stems; use `bass` stems. 45 unique masters form the "source identity" factor.

**Files:**
- Create: `src/data/moisesdb.py`
- Test: `tests/data/test_moisesdb_loader.py`

- [ ] **Step 1: Write failing test asserting item schema `(wav[1, T], {track_id, stem="bass", segment_idx})` with 4-second random-crop train mode and deterministic stride inference mode**
- [ ] **Step 2: Implement with `torchaudio.load` + on-the-fly resampling to 16 kHz + loudness normalization to −23 LUFS (via `pyln.normalize.loudness`)**
- [ ] **Step 3: Provide a `MoisesDBBass(split, root, crop_seconds, deterministic)` API**
- [ ] **Step 4: Test passes**
- [ ] **Step 5: Commit — `feat(data): MoisesDB bass-stem loader with loudness normalization`**

### Task 1.3: Zero-shot OOD set ("famous bass lines")

**Rationale:** The proposal calls out Pink Floyd "Money", etc. Build a curated, license-compliant set using stems users legally own OR commercially released isolated stems (`Mr. Bungle`, `Nathan East`, Rockschool isolated tracks, Cambridge MT Multitrack library). Each clip is labeled with dominant MIDI-pitch sequence extracted by `crepe` (Kim et al. 2018) so that content equivariance can be measured without requiring synthetic ground truth.

**Files:**
- Create: `src/data/zero_shot.py`, `scripts/prepare_zero_shot.py`, `configs/data/zero_shot_famous.yaml`
- Test: `tests/data/test_zero_shot_loader.py`

- [ ] **Step 1: Failing test asserting loader yields `(wav, {track_name, crepe_pitch_curve[T'], beat_grid})`**
- [ ] **Step 2: Implement `prepare_zero_shot.py` that (a) segments user-provided stems into 4-second windows at hop 2 s, (b) runs `crepe.predict(step_size=10)` to get pitch curve at 100 Hz, (c) writes `.pt` shards**
- [ ] **Step 3: Implement `ZeroShotBass(shard_dir)` as a `torch.utils.data.Dataset`**
- [ ] **Step 4: Test passes with a 3-clip toy fixture**
- [ ] **Step 5: Commit — `feat(data): zero-shot OOD famous-bass dataset with CREPE pitch labels`**

### Task 1.4: Log-mel spectrogram frontend (deterministic, differentiable)

**Reference:** Standard NSynth / DDSP configuration; 128 mels, 25 ms window, 10 ms hop, power-to-dB in [−80, 0].

**Files:**
- Create: `src/data/features.py`
- Test: `tests/data/test_features.py`

- [ ] **Step 1: Failing test asserting `mel(torch.randn(1, 64000)) -> (128, 401)` and idempotent re-application (shape stable)**
- [ ] **Step 2: Implement**

```python
# src/data/features.py
import torch, torchaudio

class LogMel(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=128, fmin=20, fmax=8000):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
        )
        self.a2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, wav):
        return self.a2db(self.melspec(wav)).clamp(min=-80.0, max=0.0)
```

- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(data): log-mel spectrogram frontend`**

### Task 1.5: Pitch-shift group action `T_g` (formant-preserving)

**Reference:** rubberband's `--formant` flag preserves timbre centroid; crucial so that `T_g` acts only on pitch, not timbre, making the group action well-defined on the intended factor. `torchaudio.transforms.PitchShift` uses phase-vocoder without formant preservation → **unsuitable**. We therefore wrap `pyrubberband` in a PyTorch `Dataset` transform.

**Critical test:** for randomly sampled `g` cents, the absolute RMS-level difference between `T_{−g}(T_g(x))` and `x` must be < −40 dB. If this fails the group action is not invertible and symmetry loss cannot converge.

**Files:**
- Create: `src/data/augment.py`
- Test: `tests/data/test_augment_group_action.py`

- [ ] **Step 1: Failing test**

```python
# tests/data/test_augment_group_action.py
import torch
from src.data.augment import PitchShiftGroup

def test_group_inverse_reconstructs():
    wav = torch.randn(1, 16000 * 2) * 0.1
    aug = PitchShiftGroup(sample_rate=16000, preserve_formants=True)
    g_cents = 400
    y = aug.apply(wav, g_cents)
    x_hat = aug.apply(y, -g_cents)
    err = (wav - x_hat[..., : wav.shape[-1]]).pow(2).mean().sqrt()
    assert 20 * torch.log10(err + 1e-9) < -35
```

- [ ] **Step 2: Implement using `pyrubberband.pyrb.pitch_shift(y, sr, n_steps=g/100, rbargs={"--formant": ""})` inside a numpy bridge**
- [ ] **Step 3: Expose `.sample_g(range_cents=(-1200, 1200))` returning an int divisible by 50 (semitone quanta optional)**
- [ ] **Step 4: Expose `.representation(g, d_content) -> Tensor[d_content, d_content]` stub for later Task 3.3**
- [ ] **Step 5: Test passes**
- [ ] **Step 6: Commit — `feat(data): formant-preserving pitch-shift group action`**

### Task 1.6: Collate + DataLoader builder with paired `(x, T_g x, g)` output

**Rationale:** SC-VAE training step needs triplets. Prefetching `T_g x` during `__getitem__` is CPU-expensive; do it with a `persistent_workers=True`, `num_workers=8` DataLoader.

**Files:**
- Create: `src/data/collate.py`
- Test: `tests/data/test_collate.py`

- [ ] **Step 1: Failing test asserting batch keys `{"x", "x_g", "g_cents", "labels"}` with shapes `(B,1,T)`, `(B,1,T)`, `(B,)`, dict-of-tensors**
- [ ] **Step 2: Implement `PairedPitchShiftCollate(aug, g_sampler)`**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(data): paired collate with group-action sampling`**

---

## Phase 2 — Baselines

> **Publishing discipline:** every baseline is implemented in our codebase (not third-party snapshots) to guarantee identical data pipeline, optimizer, and assessment harness. This is non-negotiable; reviewers at top venues will reject cross-codebase comparisons.

### Task 2.1: Shared convolutional encoder trunk

**Reference:** Donahue et al. 2019 (Adversarial Audio Synthesis) + NSynth baseline encoder. Stack: 5× (Conv1d stride=2, GroupNorm, SiLU) over raw waveform **OR** 5× Conv2d over log-mel. We use log-mel (lower compute, proven in NSynth) as the default and keep raw-waveform as an ablation.

**Files:**
- Create: `src/models/encoder.py`
- Test: `tests/models/test_encoder_shapes.py`

- [ ] **Step 1: Failing test asserting `Encoder(mel_shape=(128,401)) -> Tensor[B, C=512, T'=25]` deterministic under inference mode**
- [ ] **Step 2: Implement**

```python
# src/models/encoder.py
import torch, torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, stride), padding=(1, 1)),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class MelEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2)):
        super().__init__()
        c_in = 1
        self.blocks = nn.ModuleList()
        for c, s in zip(channels, strides):
            self.blocks.append(ConvBlock(c_in, c, s))
            c_in = c
        self.out_channels = channels[-1]

    def forward(self, mel):  # (B,1,128,T)
        h = mel
        for b in self.blocks:
            h = b(h)
        h = h.mean(dim=2)  # collapse freq
        return h           # (B, C, T')
```

- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(models): shared conv encoder trunk`**

### Task 2.2: β-VAE baseline

**Reference:** Higgins et al. 2017 (β-VAE); loss `L = E_q[log p(x|z)] − β·KL(q(z|x)‖p(z))`, Gaussian posterior and standard-normal prior. Set β sweep {1, 4, 8, 16}.

**Files:**
- Create: `src/models/baselines/beta_vae.py`
- Test: `tests/models/test_baselines.py::test_beta_vae_forward`

- [ ] **Step 1: Failing test asserting `BetaVAE(d_z=64)(mel)` returns `{"x_hat","mu","logvar","z"}` with correct shapes and finite values**
- [ ] **Step 2: Implement encoder → flatten → linear heads (`μ`, `log σ²`) → reparameterize → deconv decoder mirroring encoder**
- [ ] **Step 3: Provide `.loss(batch, beta) -> (total, dict)` with reconstruction as L1 on log-mel (more perceptual than MSE; see NSynth appendix)**
- [ ] **Step 4: Test passes**
- [ ] **Step 5: Commit — `feat(models): beta-VAE baseline`**

### Task 2.3: FactorVAE baseline

**Reference:** Kim & Mnih 2018, *Disentangling by Factorising* (arXiv:1802.04942). Adds a Total Correlation term `γ·KL(q(z)‖∏q(z_j))` estimated via a density-ratio discriminator `D(z)` trained adversarially on permuted minibatch samples.

**Files:**
- Create: `src/models/baselines/factor_vae.py`, `src/losses/tc.py`
- Test: `tests/losses/test_tc.py`, `tests/models/test_baselines.py::test_factor_vae`

- [ ] **Step 1: Failing test on TC estimator: on samples drawn from factored prior, `TC_density_ratio` returns value near 0 (within 0.05) after 1k discriminator steps; on correlated Gaussians with ρ=0.9, TC > 0.3**
- [ ] **Step 2: Implement permute-dim sampler, MLP discriminator (5 layers, 1000 units, LeakyReLU 0.2), WGAN-GP stabilization optional**
- [ ] **Step 3: Implement `FactorVAE` which alternates VAE step and discriminator step per minibatch half-split as in the paper**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Commit — `feat(models): FactorVAE baseline with TC discriminator`**

### Task 2.4: Autoregressive Hierarchical VAE baseline

**Reference:** Sønderby et al. 2016 (Ladder VAE) and DRAW/NVAE; for audio, the closest analog is Dieleman et al. / Engel et al. hierarchical VAE variants referenced in NSynth. Here we implement a 3-level ladder with bottom-up + top-down paths and an autoregressive Gaussian prior `p(z_L|z_{>L})`.

**Files:**
- Create: `src/models/baselines/ar_hvae.py`
- Test: `tests/models/test_baselines.py::test_ar_hvae_kl_positive`

- [ ] **Step 1: Failing test asserting all KL terms non-negative, and ELBO computed against the sum of per-level KLs**
- [ ] **Step 2: Implement 3-level ladder; top-level dim 8, mid 16, bottom 32**
- [ ] **Step 3: Tests pass**
- [ ] **Step 4: Commit — `feat(models): autoregressive hierarchical VAE baseline`**

### Task 2.5: β-TCVAE baseline

**Reference:** Chen et al. 2018, *Isolating Sources of Disentanglement in VAEs* (arXiv:1803.05428). Decomposes KL into index-code MI, Total Correlation, and dimension-wise KL; penalizes TC directly via batch-wise minibatch-weighted-sampling estimator — no discriminator. Typically stronger and simpler than FactorVAE; reviewers expect it when FactorVAE is cited.

**Files:**
- Create: `src/models/baselines/beta_tcvae.py`, `src/losses/tc.py` (extended with `batch_tc(z, mu, logvar)`)
- Test: `tests/losses/test_tc.py::test_batch_tc`, `tests/models/test_baselines.py::test_beta_tcvae`

- [ ] **Step 1: Failing test on batch-wise TC estimator: samples drawn from factored posterior give TC ≈ 0 (±0.1), correlated samples give TC > 0.3**
- [ ] **Step 2: Implement minibatch-weighted-sampling TC estimator (Eq. 4 of the paper): `log q(z) ≈ log (1/NM) Σ_m q(z_n|x_m) − log N`**
- [ ] **Step 3: Implement `BetaTCVAE` reusing β-VAE encoder/decoder; loss = recon + `α · MI + β · TC + γ · dim-wise-KL` with α=γ=1, β-sweep {1, 4, 8, 16}**
- [ ] **Step 4: Tests pass**
- [ ] **Step 5: Commit — `feat(models): beta-TCVAE baseline`**

---

## Phase 3 — Symmetry-Constrained VAE (core contribution)

### Task 3.1: Style head (invariance)

**Design:** Time-pool the encoder trunk output `(B, C, T')` with attention pooling, then two linear heads produce `(μ_s, log σ_s²) ∈ R^{d_s}`. The posterior `q(z_s|x)` is intended to be invariant to `T_g`. We will *enforce* invariance via loss L_inv (Task 3.5); the architecture itself uses attention pooling (Vaswani-style) so gradient signal can suppress time-varying leakage.

**Files:**
- Create: `src/models/style_head.py`
- Test: `tests/models/test_style_head.py`

- [ ] **Step 1: Failing test asserting `StyleHead(C=512, d_s=32)(h)` returns `{"mu_s","logvar_s","attn_weights"}` with `mu_s` shape `(B, 32)` and attention weights summing to 1**
- [ ] **Step 2: Implement learnable query vector `q`, `α_t = softmax(q^T W h_t)`, pooled `p = Σ α_t h_t`; feed to `μ = Linear(p)`, `log σ² = Linear(p)`**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(models): style head with attention pooling`**

### Task 3.2: Content head (equivariance)

**Design:** Preserve the time axis. `q(z_c|x)` is parameterized as `(μ_c[t], log σ_c²[t]) ∈ R^{d_c × T'}`. Produced by 1×1 conv heads over `(B, C, T')`.

**Files:**
- Create: `src/models/content_head.py`
- Test: `tests/models/test_content_head.py`

- [ ] **Step 1: Failing test asserting output shape `(B, d_c, T')`**
- [ ] **Step 2: Implement two `nn.Conv1d(C, d_c, 1)` heads**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(models): time-resolved content head`**

### Task 3.3: Group representation `ρ(g)`

**Reference:** Higgins et al. 2018, *Towards a Definition of Disentangled Representations* (arXiv:1812.02230). The group action `Z × R^{d_c} → R^{d_c}` can be realized as (a) a learnable block-diagonal rotation `ρ(g) = diag(R(θ_k(g)))` with `θ_k(g) = ω_k · g`, i.e. a direct sum of 2D rotations with learnable frequencies `ω_k`; (b) a learnable shift `ρ(g)z = z + g · v` for a learned direction `v ∈ R^{d_c}`. Option (a) is faithful to the cyclic structure of pitch (octave equivalence) and is our primary design; (b) is an ablation.

**Group-homomorphism property to test:** `ρ(g₁+g₂) = ρ(g₁)ρ(g₂)` and `ρ(0) = I`.

**Files:**
- Create: `src/models/group_repr.py`
- Test: `tests/models/test_group_repr.py`

- [ ] **Step 1: Failing test**

```python
# tests/models/test_group_repr.py
import torch
from src.models.group_repr import RotationRep

def test_homomorphism():
    d_c = 16  # must be even
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g1, g2 = 250.0, 475.0
    R1 = rep.matrix(torch.tensor([g1]))
    R2 = rep.matrix(torch.tensor([g2]))
    R12 = rep.matrix(torch.tensor([g1 + g2]))
    assert torch.allclose(R1 @ R2, R12, atol=1e-5)

def test_identity():
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    R0 = rep.matrix(torch.tensor([0.0]))
    assert torch.allclose(R0, torch.eye(d_c).unsqueeze(0), atol=1e-6)

def test_linearity():
    # Higgins et al. 2018 requires rho(g) to act linearly on z_c.
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g = torch.tensor([300.0])
    z1 = torch.randn(1, d_c, 10)
    z2 = torch.randn(1, d_c, 10)
    a, b = 0.7, -1.3
    lhs = rep(a * z1 + b * z2, g)
    rhs = a * rep(z1, g) + b * rep(z2, g)
    assert torch.allclose(lhs, rhs, atol=1e-5)
```

- [ ] **Step 2: Implement**

```python
# src/models/group_repr.py
import math, torch, torch.nn as nn

class RotationRep(nn.Module):
    """rho(g) = block-diag 2D rotations with learnable angular frequencies omega_k.

    For pitch shift g (in cents), block k uses angle theta_k = omega_k * (2 pi g / period).
    """
    def __init__(self, d_c: int, period_cents: float = 1200.0, freq_init: str = "octave"):
        super().__init__()
        assert d_c % 2 == 0
        self.n_blocks = d_c // 2
        self.period = period_cents
        if freq_init == "octave":
            init = torch.arange(1, self.n_blocks + 1, dtype=torch.float32)
        else:
            init = torch.ones(self.n_blocks)
        self.log_omega = nn.Parameter(torch.log(init))

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        # g_cents: (B,) -> returns (B, d_c, d_c)
        B = g_cents.shape[0]
        omega = self.log_omega.exp()                      # (K,)
        theta = (2 * math.pi / self.period) * g_cents[:, None] * omega[None, :]  # (B,K)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.zeros(B, self.n_blocks * 2, self.n_blocks * 2, device=g_cents.device)
        idx = torch.arange(self.n_blocks)
        R[:, 2 * idx, 2 * idx] = c
        R[:, 2 * idx, 2 * idx + 1] = -s
        R[:, 2 * idx + 1, 2 * idx] = s
        R[:, 2 * idx + 1, 2 * idx + 1] = c
        return R

    def forward(self, z_c: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        # z_c: (B, d_c, T')
        R = self.matrix(g_cents)                          # (B, d_c, d_c)
        return torch.einsum("bij,bjt->bit", R, z_c)
```

- [ ] **Step 3: Tests pass**
- [ ] **Step 4: Commit — `feat(models): rotational group representation rho(g)`**

### Task 3.4: Decoder

**Design:** Symmetric to the encoder with `ConvTranspose2d`; input is a tile of `z_s` broadcast along time concatenated with `z_c` along the channel dim, producing `(B, d_s + d_c, 1, T')`, upsampled to `(B, 1, 128, T)` log-mel.

**Files:**
- Create: `src/models/decoder.py`
- Test: `tests/models/test_decoder_shapes.py`

- [ ] **Step 1: Failing test asserting decoder maps `(B, d_s) × (B, d_c, T')` back to mel shape**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(models): decoder mirroring encoder`**

### Task 3.5: Symmetry losses

**Math (must match code exactly):**

Style invariance:
$$\mathcal{L}_{\text{inv}} = \mathbb{E}_{x, g}\left[\big\|\mu_s(x) - \mu_s(T_g x)\big\|_2^2\right]$$

Content equivariance:
$$\mathcal{L}_{\text{equi}} = \mathbb{E}_{x, g}\left[\big\|\mu_c(T_g x) - \rho(g)\,\mu_c(x)\big\|_2^2\right]$$

Cross-swap consistency (novel, reviewer-bait):
$$\mathcal{L}_{\text{swap}} = \mathbb{E}_{x, g}\left[\big\| \text{Dec}(z_s(x), \rho(g) z_c(x)) - T_g x \big\|_1\right]$$

Final composite (per-subspace KL — prevents posterior collapse in the constrained subspace while `β_c` can stay low):
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta_s \cdot \mathrm{KL}(q(z_s|x)\|p(z_s)) + \beta_c \cdot \mathrm{KL}(q(z_c|x)\|p(z_c)) + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}} + \lambda_{\text{equi}} \mathcal{L}_{\text{equi}} + \lambda_{\text{swap}} \mathcal{L}_{\text{swap}}$$

with `β_s, β_c ∈ [1, 8]` independently (cyclical anneal; Fu et al. 2019), `λ_inv ∈ {0.1, 1, 10}`, `λ_equi ∈ {0.1, 1, 10}`, `λ_swap ∈ {0, 0.5, 1}`. Decomposed KL components (Total Correlation, index-code MI, dimension-wise KL per β-TCVAE batch estimator, Chen et al. 2018, arXiv:1803.05428) are logged per-subspace to W&B for diagnostic insight — not optimized directly.

**Files:**
- Create: `src/losses/symmetry.py`
- Test: `tests/losses/test_symmetry.py`

- [ ] **Step 1: Failing test asserting (a) `L_inv=0` when `mu_s(x)==mu_s(T_g x)`, (b) `L_equi=0` when `mu_c(T_g x) = rho(g) mu_c(x)` for a synthetic rep, (c) both losses are strictly positive on random tensors**
- [ ] **Step 2: Implement**

```python
# src/losses/symmetry.py
import torch, torch.nn.functional as F

def invariance_loss(mu_s_x, mu_s_gx):
    return (mu_s_x - mu_s_gx).pow(2).sum(dim=-1).mean()

def equivariance_loss(mu_c_gx, rho_g_mu_c_x):
    return (mu_c_gx - rho_g_mu_c_x).pow(2).sum(dim=1).mean()

def swap_consistency_loss(decoded_swapped_mel, target_mel):
    return F.l1_loss(decoded_swapped_mel, target_mel)
```

- [ ] **Step 3: Tests pass**
- [ ] **Step 4: Commit — `feat(losses): invariance, equivariance, swap-consistency`**

### Task 3.6: KL loss with free-bits and cyclical β annealing

**Reference:** Kingma 2016 (IAF, free-bits), Fu et al. 2019 (cyclical annealing). Free-bits nat threshold `τ_s = τ_c = 0.1` per latent group, applied *separately* to `z_s` and `z_c`. Return separate scalars `kl_s, kl_c` so `training_step` can weight them by `β_s, β_c` and log them independently.

**Files:**
- Create: `src/losses/kl.py`
- Test: `tests/losses/test_kl.py`

- [ ] **Step 1: Failing tests: (a) analytic Gaussian KL for `N(0,I) vs N(0,I)` equals 0, (b) free-bits with τ=0.5 clamps KL at ≥0.5 nats/dim, (c) cyclical β schedule returns 1.0 at cycle end and 0.0 at cycle start, (d) `kl_subspace(mu_s, logvar_s)` and `kl_subspace(mu_c, logvar_c)` return independent scalars with independent free-bit clamps**
- [ ] **Step 2: Implement `kl_subspace(mu, logvar, free_bits)` and a top-level `kl_total(mu_s, logvar_s, mu_c, logvar_c, tau_s, tau_c)` returning `(kl_s, kl_c)`**
- [ ] **Step 3: Tests pass**
- [ ] **Step 4: Commit — `feat(losses): per-subspace KL with free-bits and cyclical beta annealing`**

### Task 3.7: Multi-scale spectral reconstruction loss

**Reference:** Yamamoto et al. 2020 (Parallel WaveGAN), Engel et al. 2020 (DDSP §3.2). Sums log-magnitude STFT errors across FFT sizes {2048, 1024, 512, 256, 128, 64}, L1-norm per scale. We apply it on the *decoded mel-spectrogram converted to linear magnitude via inverse mel* — but since we decode mel directly, we use (a) L1 on log-mel (fast) + (b) an auxiliary multi-scale STFT on the HiFi-GAN-reconstructed waveform computed every 5 epochs (too expensive every step).

**Files:**
- Create: `src/losses/recon.py`
- Test: `tests/losses/test_recon.py`

- [ ] **Step 1: Failing test: on identical inputs loss = 0; positive on shuffled**
- [ ] **Step 2: Implement with `torchaudio.transforms.Spectrogram` at each scale**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(losses): multi-scale spectral + L1-mel reconstruction`**

### Task 3.8: Full SC-VAE module

**Files:**
- Create: `src/models/sc_vae.py`
- Test: `tests/models/test_sc_vae_forward.py`

- [ ] **Step 1: Failing test asserting `SCVAE(cfg).forward(batch) -> dict` with keys `{x_hat, x_g_hat, mu_s_x, mu_s_gx, mu_c_x, mu_c_gx, rho_g_mu_c_x, ...}` and `.loss(batch) -> (total_scalar, components_dict)`**
- [ ] **Step 2: Implement, wiring StyleHead + ContentHead + RotationRep + Decoder**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(models): full SC-VAE assembly`**

---

## Phase 4 — Training Infrastructure

### Task 4.1: Lightning module wrapping any model via config

**Files:**
- Create: `src/training/lit_module.py`
- Test: `tests/training/test_lit_module.py` (smoke train-one-step on a tiny random dataset)

- [ ] **Step 1: Failing test running `Trainer(fast_dev_run=True).fit(module, loader)` with dummy dataset of 8 items**
- [ ] **Step 2: Implement. Key points: (a) `training_step` computes composite loss and logs every component separately: `recon`, `kl_s`, `kl_c`, `L_inv`, `L_equi`, `L_swap`, plus β-TCVAE decomposition per-subspace (`tc_s`, `mi_s`, `dimwise_kl_s`, same for `_c`) as diagnostics only (not optimized); (b) `configure_optimizers` returns AdamW (lr 1e-4, betas (0.9, 0.99), weight_decay 1e-6) + cosine schedule with 5k warmup; (c) EMA of weights with decay 0.999 via callback**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(training): Lightning module`**

### Task 4.2: Hydra entrypoint

**Files:**
- Create: `src/training/cli.py`, all config YAMLs under `configs/`

- [ ] **Step 1: Write minimal config for `configs/train/base.yaml`**
- [ ] **Step 2: Implement `@hydra.main` CLI that instantiates `data`, `model`, `trainer`**
- [ ] **Step 3: Smoke-run `python -m src.training.cli +trainer.fast_dev_run=true` and assert exit code 0 via test**
- [ ] **Step 4: Commit — `feat(training): hydra CLI entrypoint`**

### Task 4.3: W&B logging + audio reconstruction artifacts every epoch

- [ ] **Step 1: Failing test asserting logger is registered and `on_validation_epoch_end` would log 4 sample reconstructions (checked by counting calls to a mock)**
- [ ] **Step 2: Implement callback `AudioReconCallback(num_samples=4)` that (a) picks fixed validation batch, (b) decodes mel → waveform via pretrained BigVGAN, (c) logs wandb.Audio**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(training): wandb audio reconstruction logger`**

### Task 4.4: Gradient-flow and NaN guards

- [ ] **Step 1: Failing test asserting module raises when encoder output contains NaN**
- [ ] **Step 2: Implement `torch.autograd.set_detect_anomaly(cfg.debug)` guard + `clip_grad_norm_(1.0)` per Lightning config**
- [ ] **Step 3: Commit**

---

## Phase 5 — Assessment Metrics

### Task 5.1: MIG (Mutual Information Gap)

**Reference:** Chen et al. 2018, *Isolating Sources of Disentanglement in VAEs* (arXiv:1802.04942). For each ground-truth factor `v_k`, discretize latents per-dim into 20 equal-frequency bins, estimate `I(z_i; v_k)` via empirical joint histogram, then `MIG = (1/K) Σ_k (I(z_{i*}; v_k) − I(z_{i**}; v_k)) / H(v_k)` where `i*` is the top latent and `i**` the second-best.

**Files:**
- Create: `src/evaluation/mig.py`
- Test: `tests/evaluation/test_mig.py`

- [ ] **Step 1: Failing test with synthetic data where `z_0 = v_0 + ε_small, z_1 = v_1 + ε_small, z_{2..} = noise`. Expect MIG > 0.7.**
- [ ] **Step 2: Implement with sklearn `mutual_info_classif` for discrete factors, and a histogram-based MI for continuous factors (e.g. pitch treated as 88-bin discrete)**
- [ ] **Step 3: Provide `compute(model, dataloader, factor_keys=["pitch","instrument_id"])`**
- [ ] **Step 4: Test passes**
- [ ] **Step 5: Commit — `feat(eval): MIG metric`**

### Task 5.2: DCI

**Reference:** Eastwood & Williams 2018, *A Framework for the Quantitative Evaluation of Disentangled Representations*. Fit per-factor Gradient Boosted Trees `v_k = f_k(z)`, compute feature importances `R[i,k]`. Then:
- Disentanglement per dim: `D_i = 1 − H_K(P_{i·}); P_{ik} = R_{ik}/Σ_{k'}R_{ik'}`; overall `D = Σ_i ρ_i D_i` with `ρ_i = Σ_k R_{ik}/ΣR`.
- Completeness per factor: `C_k = 1 − H_d(P_{·k}); P_{ik} = R_{ik}/Σ_{i'}R_{i'k}`.
- Informativeness: reported RMSE (or accuracy) of predicting `v_k` from `z` on held-out set.

**Files:**
- Create: `src/evaluation/dci.py`
- Test: `tests/evaluation/test_dci.py`

- [ ] **Step 1: Failing test with synthetic data as in Task 5.1, expect `D > 0.8, C > 0.8, I > 0.8`**
- [ ] **Step 2: Implement with `sklearn.ensemble.GradientBoostingRegressor/Classifier`, `max_depth=6, n_estimators=500`**
- [ ] **Step 3: Return pandas DataFrame + scalar summaries**
- [ ] **Step 4: Test passes**
- [ ] **Step 5: Commit — `feat(eval): DCI framework`**

### Task 5.3: SRR (Signal-to-Reconstruction Ratio)

**Definition:**
$$\text{SRR}(x, \hat{x}) = 10 \log_{10} \frac{\sum|S(x)|^2}{\sum|S(x) - S(\hat{x})|^2}$$
with `S` = magnitude STFT averaged over window sizes {512, 1024, 2048}.

**Files:** Create `src/evaluation/srr.py`; Test `tests/evaluation/test_srr.py`.

- [ ] **Step 1: Failing test: `SRR(x, x) = inf` (return `torch.inf`), `SRR(x, noise)` small**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Test passes**
- [ ] **Step 4: Commit — `feat(eval): SRR metric`**

### Task 5.4: Equivariance/Invariance diagnostic

**Metric:** empirical invariance ratio `IR = ‖μ_s(x)−μ_s(T_g x)‖₂² / ‖μ_s(x)‖₂²` averaged over assessment set and `g ∈ {±100, ±200, …, ±1200} cents`. Equivariance ratio `ER = ‖μ_c(T_g x)−ρ(g)μ_c(x)‖₂² / ‖μ_c(T_g x)‖₂²`.

**Files:** Create `src/evaluation/equivariance_err.py`; test asserts metrics drop for a mocked perfectly-equivariant model.

- [ ] Steps 1–4 analogous; commit `feat(eval): invariance and equivariance diagnostics`.

### Task 5.5: Latent traversal + identity-swap audio artifacts

**Files:** Create `src/evaluation/latent_traversal.py`.

- [ ] **Step 1:** Generate grids: (a) vary `z_s` across two NSynth instruments while holding `z_c` fixed → expect only timbre change; (b) vary `g` along `ρ(g)z_c` while holding `z_s` fixed → expect only pitch change; (c) **spherical interpolation (slerp)** between two `z_s` vectors from different instruments while holding `z_c` fixed — render audio at 8 interpolation steps to evaluate smoothness of timbre manifold (NSynth, Engel et al. 2017, §4.2). Save WAVs + mel PNGs to `runs/<exp>/qualitative/`.
- [ ] **Step 2:** Commit — `feat(eval): latent traversal artifacts with timbre slerp`.

### Task 5.6: SAP + Modularity + FactorVAE score

**References:** Seetharaman et al. 2020 (arXiv:2110.05587) — SAP for interdependent attributes; Ridgeway & Mozer 2018 — Modularity; Kim & Mnih 2018 (arXiv:1802.04942) — FactorVAE score via majority-vote classifier. MIG alone is insufficient when factors are correlated (Pati & Lerch 2021, arXiv:2108.01450).

**Files:**
- Create: `src/evaluation/sap.py`, `src/evaluation/modularity.py`, `src/evaluation/factor_vae_score.py`
- Test: `tests/evaluation/test_sap.py`, `tests/evaluation/test_modularity.py`, `tests/evaluation/test_factor_vae_score.py`

- [ ] **Step 1: Failing SAP test:** synthetic dataset where `z_0 ≈ v_0, z_1 ≈ v_1, rest=noise` yields `SAP > 0.5`.
- [ ] **Step 2: Implement SAP:** train a linear SVC per (latent-dim, factor) pair; `SAP_k = top1_acc − top2_acc` over dims; average over `k`. For continuous factors (pitch) use `LinearRegression` R².
- [ ] **Step 3: Failing Modularity test:** synthetic data as above yields `Mod > 0.8`.
- [ ] **Step 4: Implement Modularity (Ridgeway & Mozer):** for each dim `i`, compute normalized MI with each factor `k`, then `Mod_i = 1 − (Σ_k θ_{ik}² − θ_max²) / (θ_max²·(K−1))`.
- [ ] **Step 5: Implement FactorVAE score:** fix factor `k`, sample batch with constant `v_k`, compute dim-wise empirical variances of `z`, identify argmin dim; train majority-vote classifier on (argmin-dim, k) pairs; report classifier accuracy.
- [ ] **Step 6: Tests pass**
- [ ] **Step 7: Commit — `feat(eval): SAP, Modularity, FactorVAE-score metrics`**

### Task 5.7: Task-based perceptual eval — timbre-swap pitch accuracy

**Reference:** Pati & Lerch 2021 (arXiv:2108.01450) — task-level controllability eval beats metric-only eval. Probes that `z_c` actually carries pitch and is decodable to correct pitch under a swapped timbre.

**Files:**
- Create: `src/evaluation/swap_pitch_acc.py`
- Test: `tests/evaluation/test_swap_pitch_acc.py`

- [ ] **Step 1:** Pin CREPE (Kim et al. 2018) as pitch extractor via `torchcrepe` (CC-BY-NC-SA noted). On clean NSynth notes, `|f0_hat − f0_gt| < 20 cents` on ≥95% of frames — test this on fixture.
- [ ] **Step 2:** Define protocol — sample `(x_a, x_b)` with different instruments and different pitches from the test set; render `x̂ = HiFiGAN(Dec(z_s(x_b), z_c(x_a)))`; extract `f0(x̂)` via CREPE; compare to `f0(x_a)`; define `PitchAcc@50cents` = fraction of frames within 50 cents.
- [ ] **Step 3:** Run on 500 pairs per model; report mean ± bootstrap CI95.
- [ ] **Step 4:** Commit — `feat(eval): timbre-swap pitch accuracy via CREPE`.

### Task 5.8: Aggregate assessment report

**Files:** `src/evaluation/report.py`, `scripts/run_full_eval.sh`.

- [ ] **Step 1:** Generate LaTeX table `results.tex` with rows {β-VAE, β-TCVAE, FactorVAE, AR-HVAE, SC-VAE} × columns {MIG, SAP, Modularity, FactorVAE-score, DCI-D, DCI-C, DCI-I (RMSE for pitch, acc for instrument), SRR, IR, ER, PitchAcc@50}. Include mean ± std across 5 seeds. Fix DCI predictor hyperparameters (`max_depth=6, n_estimators=500`) per Eastwood & Williams 2018 for reproducibility.
- [ ] **Step 2:** Emit DCI feature-importance matrix `R` as a heatmap PNG per model (rows = latent dims, columns = factors) to `docs/figures/dci_heatmap_<model>.pdf`.
- [ ] **Step 3:** Emit Pareto-front scatter (x = SRR, y = SAP) over *all* runs of the hyperparameter sweep for every model — visualizes disentanglement/reconstruction tradeoff (Locatello et al. 2019). Save `docs/figures/pareto_srr_sap.pdf`.
- [ ] **Step 4:** Commit — `feat(eval): aggregate report with Pareto front and DCI heatmaps`.

---

## Phase 6 — Experimental Protocol (publication-grade)

> Reviewers at NeurIPS/ICML/ICLR expect: (i) ≥5 random seeds, (ii) statistically significant deltas (paired t-test p < 0.01), (iii) identical compute budget across baselines, (iv) full hyper-parameter sweep disclosure, (v) released artifacts.

### Task 6.1: Seed management and reproducibility harness

- [ ] Add `src/utils/seed.py` with `set_seeds(seed)` touching `random, numpy, torch, torch.cuda, torch.backends.cudnn.deterministic=True, benchmark=False`.
- [ ] Each run writes `run_manifest.json` containing git commit SHA, config dump, pip freeze, CUDA/cuDNN versions, hardware info.
- [ ] Commit — `feat(utils): deterministic seed + run manifest`.

### Task 6.2: Hyper-parameter sweep configuration

**Sweep space (disclosed in appendix):**
- `d_s ∈ {16, 32, 64}`, `d_c ∈ {16, 32, 64}` with `d_c` even.
- `β_s` schedule: constant ∈ {1, 4, 8} ∪ cyclical (4 cycles to β_s=4).
- `β_c` schedule: constant ∈ {1, 4, 8} ∪ cyclical (4 cycles to β_c=4) — swept independently from `β_s`.
- `λ_inv, λ_equi ∈ {0, 0.1, 1, 10}` (zero-case verifies losses are responsible for gains).
- `λ_swap ∈ {0, 0.5, 1}`.
- `g_cents` sampling distribution ∈ {uniform(−1200,1200), uniform(−600,600), discrete semitones}.
- `ρ` parameterization ∈ {rotation, translation, identity} (identity is ablation).

Write `configs/train/sweep.yaml` with a Hydra multirun + W&B sweep YAML.

- [ ] Commit — `chore(train): hyper-parameter sweep configuration`.

### Task 6.3: Ablation matrix

Run the following ablations on NSynth-bass (single seed) then re-run top-3 on MoisesDB (5 seeds):

| Ablation | Purpose |
|---|---|
| SC-VAE full | Main claim |
| − L_inv | Does invariance loss matter? |
| − L_equi | Does equivariance loss matter? |
| − L_swap | Reviewer robustness question |
| ρ = identity | Shows ρ is what buys equivariance, not just paired inputs |
| Replace rubberband with phase-vocoder (no formant preservation) | Shows group-action fidelity matters |
| Raw-waveform encoder (1D dilated conv, WaveNet-style) | Frontend ablation |
| DDSP decoder | Decoder class ablation |
| Swap `z_s`/`z_c` dims 16/64 → 64/16 | Capacity ablation |
| No paired batch (use different minibatches for `x`, `T_g x`) | Shows pairing is necessary |
| β fixed vs cyclical | Training stability ablation |
| Synthetic pre-training on oscillator+filter synth | Does simulation-as-supervision (arXiv:2505.23305) improve small-data MIG/SAP? |
| Pitch-scalar `g` vs learned embedding `φ(g)` | Does learned non-linear embedding of transformation improve equivariance (arXiv:1809.07600 analog)? |
| β_s, β_c joint sweep at matched total-KL | Does per-subspace β vs global β change posterior collapse profile? |
| FactorVAE-score / SAP / Modularity vs MIG-only selection | Does metric choice flip model ranking? (Pati & Lerch 2021) |

- [ ] Write `scripts/run_ablations.sh` iterating the above and writing results to `runs/ablations/`.
- [ ] Commit — `exp: ablation runner`.

### Task 6.4: Statistical significance

- [ ] For each metric in the main table, compute paired bootstrap (10k resamples) CI95 across seeds. Report `Δ`, `p` (paired t-test), and Cliff's delta. Add to `report.py`.
- [ ] Commit — `feat(eval): bootstrap significance in report`.

### Task 6.5: Zero-shot OOD assessment

- [ ] Run all trained checkpoints on the `ZeroShotBass` set. For each clip, (a) extract `z_s` over sliding 4-s windows and compute per-clip variance of `z_s` (lower = better identity stability); (b) shift clip by `g` and measure `ER`. Report mean variance and ER vs. baselines.
- [ ] Commit — `exp: zero-shot OOD assessment`.

### Task 6.6: Compute budget logging

- [ ] Every run records `wall_time_hours, gpu_hours, peak_vram_gb, flops_per_step` (via `fvcore`) in the manifest.
- [ ] Final appendix table summarizes budget per method; this is required to rule out "more compute" explanations.
- [ ] Commit — `feat(utils): compute budget logging`.

---

## Phase 7 — Optional Decoder Ablations (high-impact)

### Task 7.1: DDSP decoder arm

**Reference:** Engel et al. 2020 (arXiv:2001.04643). Replaces the mel decoder with a differentiable harmonic-plus-noise synthesizer. Conditioning mapping (makes DDSP inductive bias respect the symmetry split):
- `z_c` (time-resolved, equivariant) → 1D-conv heads predict normalized `f0(t) ∈ [0,1]` (sigmoid × 2kHz range) and loudness envelope `ℓ(t)` — these drive the harmonic oscillator bank.
- `z_s` (time-pooled, invariant) → MLP head predicts static harmonic-amplitude distribution `α ∈ R^{K}` (K=100 partials, softmax normalized) **and** FIR filter magnitudes for the filtered-noise synth (65 taps). Timbre stays invariant to pitch shift because `z_s` is not conditioned on time or `g`.

- [ ] **Step 1** Implement `DDSPHead`: `z_c → (f0, loudness)` via two 1D convs; `z_s → (harmonic_dist, noise_filter)` via 2-layer MLP.
- [ ] **Step 2** Implement harmonic oscillator bank (differentiable additive synth, anti-aliased) + filtered-noise synth. Sum.
- [ ] **Step 3** Losses: multi-scale STFT L1 on waveform directly (FFT sizes {2048, 1024, 512, 256, 128, 64}).
- [ ] **Step 4** Test DDSP forward produces shape `(B, 1, 64000)` and gradient flows through `f0` and `α`.
- [ ] **Step 5** Commit — `feat(models): DDSP decoder ablation arm`.

### Task 7.2: Diffusion decoder arm

**Reference:** Ho et al. 2020 (DDPM, arXiv:2006.11239) + Nichol & Dhariwal 2021 (iDDPM, arXiv:2102.09672). Conditional mel-spectrogram diffusion U-Net (1000 DDPM training steps, 50-step DDIM sampling at eval), ε-parameterization. Conditioning:
- `z_s` → added to timestep sinusoidal embedding via linear projection (FiLM-style global conditioning).
- `z_c` → concatenated to the U-Net input along the channel dim, broadcast along frequency axis (spatial conditioning preserves time structure needed for equivariance).
- Classifier-free guidance: dropout conditioning with `p=0.1` during training; at inference, sample with guidance scale `w ∈ {1, 3, 7}` and pick best via MIG on val set.

- [ ] **Step 1** Implement `DiffusionDecoder` with cosine β schedule, U-Net from `diffusers` with channel-mult `(1,2,4,4)` and attention at res 16.
- [ ] **Step 2** Integrate into `SCVAE` as a swappable decoder. KL path stays per-subspace; recon loss becomes `L_simple = E[‖ε − ε_θ(x_t, t, z_s, z_c)‖²]`.
- [ ] **Step 3** Implement DDIM 50-step sampler; measure **Real-Time Factor (RTF)** = `inference_wall_time / audio_duration` and log to manifest (per DITTO-2, Wang et al. 2024, arXiv:2211.11695 — inference latency is a first-class metric for diffusion audio).
- [ ] **Step 4** Tests on shape + toy 100-step overfit.
- [ ] **Step 5** Commit — `feat(models): diffusion decoder arm with classifier-free guidance and RTF logging`.

### Task 7.3: Synthetic pre-training arm

**Reference:** Self-Supervised Causal Representation Learning with Synthetic Data (arXiv:2505.23305) — simulation-as-supervision for causal factor recovery.

- [ ] **Step 1** Implement `src/data/synth_pretrain.py`: sinusoidal + sawtooth oscillator + biquad filter + ADSR envelope → generates `(x, f0, timbre_id)` triples on the fly. Factor structure: pitch is continuous, timbre is a categorical over 16 filter/ADSR presets.
- [ ] **Step 2** Pre-train SC-VAE 20 epochs on synthetic data only, then fine-tune on NSynth-bass for 80 epochs (same total budget as from-scratch).
- [ ] **Step 3** Measure MIG, DCI, SAP delta vs from-scratch — hypothesis: pretrain > scratch on small-data regime (MoisesDB 240 stems).
- [ ] **Step 4** Commit — `feat(models): synthetic-data pre-training arm`.

---

## Phase 8 — Paper Deliverables

### Task 8.1: Figures

- [ ] `figures/fig1_architecture.pdf` (Tikz): encoder → style/content heads → ρ(g) → decoder, with loss arrows.
- [ ] `figures/fig2_latent_traversal.pdf`: 2×4 grid of mel-spectrogram traversals with audio files as ancillary.
- [ ] `figures/fig3_equivariance_curve.pdf`: ER vs `g_cents` for all methods.
- [ ] `figures/fig4_zero_shot.pdf`: z_s stability on famous bass lines.
- [ ] Commit — `docs(paper): figures`.

### Task 8.2: Main results table

- [ ] `docs/paper/tables/main_results.tex` emitted by `report.py`.
- [ ] Commit — `docs(paper): main results table`.

### Task 8.3: Reproducibility appendix

Must cover, in order: datasets & splits, preprocessing, architecture dims per module, hyper-parameters for every run, training hardware, seed list, total GPU-hours, license notes (NSynth CC-BY-4.0, MoisesDB license), link to code+checkpoints.

- [ ] Commit — `docs(paper): reproducibility appendix`.

### Task 8.4: Broader impacts + limitations section

Required by NeurIPS. Topics: (a) misuse risk of high-fidelity timbre transfer (audio deepfakes), (b) limitation to monophonic bass (not polyphonic), (c) Western-tuned 12-TET bias, (d) dataset demographic skew of MoisesDB stems.

- [ ] Commit — `docs(paper): limitations and broader impacts`.

---

## Phase 9 — Release Engineering

- [ ] **Task 9.1** CI: GitHub Actions runs `ruff + pytest` on every PR. Commit `ci: add linting + test workflow`.
- [ ] **Task 9.2** Docker image pinning CUDA 12.1 + PyTorch 2.3.1. Commit `chore: release Dockerfile`.
- [ ] **Task 9.3** Hugging Face Hub model card with sample reconstructions. Commit `docs: HF model card`.
- [ ] **Task 9.4** Zenodo DOI for code snapshot. Commit `docs: release metadata`.

---

## Appendix A — Literature Mapping

| Module | Primary references |
|---|---|
| VAE objective | Kingma & Welling 2013 (arXiv:1312.6114) |
| β-penalty | Higgins et al. 2017 (β-VAE) |
| Total Correlation & FactorVAE | Kim & Mnih 2018 (arXiv:1802.04942); Chen et al. 2018 (β-TCVAE, MIG) |
| Disentanglement impossibility caveat | Locatello et al. 2019 (arXiv:1811.12359) — motivates supervised assessment and paired data |
| Symmetry-based definition of disentanglement + group-equivariant latents | Higgins et al. 2018 (arXiv:1812.02230) |
| DCI assessment framework | Eastwood & Williams 2018 |
| GAN decoder references | Goodfellow et al. 2014 (arXiv:1406.2661) |
| NSynth dataset + WaveNet autoencoder baseline | Engel et al. 2017 (arXiv:1704.01279) |
| DDSP differentiable synth | Engel et al. 2020 (arXiv:2001.04643) |
| Jukebox (large-scale audio VQ-VAE; conceptual precedent) | Dhariwal et al. 2020 (arXiv:2005.00341) |
| Diffusion decoder | Ho et al. 2020 (arXiv:2006.11239) |
| MusicGen (token-LM audio generation) | Copet et al. 2023 (arXiv:2306.05284) — contextual related work; positions SC-VAE as continuous/interpretable alternative to token-AR paradigm |
| β-TCVAE + MIG metric + KL decomposition | Chen et al. 2018 (arXiv:1803.05428) — primary baseline + logged diagnostic |
| Learned representation of edit/transformation | Yin et al. 2018 (arXiv:1809.07600) — motivates `φ(g)` learned-embedding ablation |
| Disentanglement ≠ controllability; task-based perceptual eval | Pati & Lerch 2021 (arXiv:2108.01450) — motivates Task 5.7 (CREPE pitch accuracy) |
| SAP metric + interdependent-attribute critique of MIG | Seetharaman et al. 2020 (arXiv:2110.05587) — SAP implementation in Task 5.6 |
| Inference latency / RTF for audio diffusion | DITTO-2, Wang et al. 2024 (arXiv:2211.11695) — RTF logging in Task 7.2 |
| MIDI-VAE (symbolic-domain disentanglement) | Brunner et al. 2018 (arXiv:2405.20289) — related-work analogy for symbolic vs audio |
| DISMIX (mixture-level pitch/timbre disentanglement) | Luo et al. 2024 (arXiv:2408.10807) — closest recent competitor; novelty contrast cited in Goal; mixture extension is future work |
| Simulation-as-supervision for causal factor learning | arXiv:2505.23305 — motivates synthetic-pretrain arm in Task 7.3 |

---

## Appendix B — Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Posterior collapse on `z_s` when `L_inv` forces constant output | Free-bits (τ=0.1) on `z_s`; monitor `KL(q(z_s|x)‖p(z_s))` and abort if < 0.2 nats |
| Group action not invertible due to STFT phase distortion | Task 1.5 invertibility test must pass; otherwise fall back to spectrogram-domain shift with magnitude-only loss |
| `ρ(g)` learns identity map (trivial equivariance) | Add ablation `ρ=I` as baseline; regularize `‖ω−ω_init‖` upward slightly |
| MIG/DCI metric variance high on small assessment set | Use ≥ 5000 examples; report 5-seed mean + bootstrap CI |
| Reviewers claim "pairing leaks labels" | Show same gains on an unpaired variant where `T_g` is applied as consistency regularizer within-batch rather than across-batch |
| Monophonic scope criticism | Preempt with limitations section; show preliminary polyphonic mixture extension as future work, pointing to DISMIX (arXiv:2408.10807) as the mixture-domain reference |
| Reviewer: "DISMIX already does disentanglement on audio" | Goal section sharpens contrast: SC-VAE = formal group-theoretic framework (Higgins 2018) with rotational `ρ(g)`; DISMIX = mixture-supervision w/o group action. Different contributions. Add head-to-head row on monophonic subset in supplementary. |
| Locatello impossibility raised by reviewer | Inductive bias is declared: known pitch-shift group structure + paired (x, T_g x) supervision — this is the supervision that breaks the impossibility result. Cite §4 of the proposal. |
| MIG ranks models differently from SAP/Modularity | Report all four metrics (MIG, SAP, Modularity, FactorVAE score) in main table; discuss divergences. Pati & Lerch 2021 predicts this. |

---

## Appendix C — Test Matrix Summary

Every task above has a failing-test-first step. Cumulative tests expected:

- `tests/data`: 7 files, ~35 assertions (adds `test_synth_pretrain.py`)
- `tests/models`: 10 files, ~48 assertions (adds `test_beta_tcvae.py`, DDSP head, diffusion decoder, linearity test for `ρ(g)`)
- `tests/losses`: 4 files, ~20 assertions (extends `test_tc.py` with batch-TC, `test_kl.py` with per-subspace free-bits)
- `tests/evaluation`: 9 files, ~55 assertions (adds `test_sap.py`, `test_modularity.py`, `test_factor_vae_score.py`, `test_swap_pitch_acc.py`)
- `tests/training`: 3 files, ~10 assertions

Target: ≥ 85% line coverage on `src/`, enforced by `pytest --cov-fail-under=85` in CI.

---

## Self-Review Checklist (already run)

- **Spec coverage:** every item in the proposal (MIG, DCI, SRR, NSynth bass, MoisesDB 240 bass stems, zero-shot famous bass, β-VAE baseline, AR-Hierarchical-VAE baseline, composite loss `L_recon + β·KL + λ·L_sym`, symmetry penalty `‖z_s(x) − z_s(T_g x)‖²`, Self-Supervised Causal Framework, group-theory constraints, pitch-shift equivariance) is mapped to a phase/task above. FactorVAE + β-TCVAE are added as extra baselines — reviewers always ask for both. Post-literature-review additions (Pass 2, 2026-04-20): (i) per-subspace KL (`β_s`, `β_c`), (ii) β-TCVAE baseline + KL decomposition diagnostic, (iii) SAP + Modularity + FactorVAE score metrics, (iv) CREPE-based timbre-swap pitch-accuracy eval, (v) slerp timbre interpolation, (vi) linearity test for `ρ(g)`, (vii) DDSP conditioning specification, (viii) classifier-free guidance + RTF for diffusion arm, (ix) synthetic pre-training arm, (x) Pareto-front and DCI-heatmap figures, (xi) DISMIX/Higgins-2018/Locatello contrast in Goal. All driven by literature review of 19 papers.
- **Placeholder scan:** clean.
- **Type consistency:** `RotationRep.forward(z_c, g_cents)` returns `(B, d_c, T')`; `equivariance_loss` accepts the same shape; `StyleHead` outputs `(B, d_s)`; `invariance_loss` sums over `dim=-1`; `kl_total(...)` returns tuple `(kl_s, kl_c)` consumed by `training_step` with separate `β_s`, `β_c`. Consistent.

---

**End of plan.**
