# Adarsh — Progress Log

## Branch: adarsh/data-pipeline

---

## April 17, 2026

### Done
- Set up musicvae conda env (Python 3.9)
- Installed core dependencies: tensorflow, librosa, numpy, note-seq, torch, torchaudio
- Cloned MusicVAE (Magenta) repo
- Cloned team repo, created and pushed adarsh/data-pipeline branch
- Set up project folder structure (data/, src/, notebooks/, results/)
- NSynthBass dataset class (src/data/nsynth_dataset.py)
- MoisesDB dataset class (src/data/moisesdb_dataset.py)
- Combined dataloader (src/data/dataloader.py)
- MIG metric (src/evaluation/mig.py)
- DCI metric (src/evaluation/dci.py)
- SRR metric (src/evaluation/srr.py)
- Evaluation runner (src/evaluation/evaluate.py)
- CNN Encoder with style + content heads (src/models/encoder.py)
- CNN Decoder with interpolation (src/models/decoder.py)
- SymmetryVAE full model + loss (src/models/vae.py)
- Training loop with grad clipping, checkpointing (train.py)
- requirements.txt
- Fixed NSynthBass to parse pitch from filename (no json needed)
- NSynth bass subset loaded: 12,075 samples
- MoisesDB download running as SLURM job (6507167)
- Conda env working: /scratch/at7095/conda_envs/musicvae (Python 3.9)
- HPC repo at: /scratch/at7095/ML_Project/Disentangled-Representations-Learning-for-Music-Processing

## April 18, 2026

### Done
- Fixed home dir quota on HPC (cleared 5.3GB from .local)
- Conda env working: /scratch/at7095/conda_envs/musicvae (Python 3.9)
- NSynth fully extracted: 65,474 bass files
- MoisesDB fully extracted: 240 tracks, 237 bass files, 12,510 chunks
- Preprocessing script: converted all audio to mel spectrograms (.npy)
  - NSynth: 65,474 npy files
  - MoisesDB: 12,510 npy files (4s chunks)
- Preprocessed dataset classes (nsynth_preprocessed.py, moisesdb_preprocessed.py)
- Fixed GPU utilization issue — npy loading keeps GPU busy
- Training job running: job 6546023, epoch 4+ in progress
- MoisesDB dataset class updated to match actual path structure

## April 18, 2026 (continued)
### Done
- Diagnosed NaN loss explosion in training job 6558198 (all 50 epochs NaN)
- Root cause: BatchNorm2d in encoder producing NaN on near-silent mel batches
- Fix: replaced all BatchNorm2d → InstanceNorm2d(affine=True) in encoder.py
- Fixed encoder forward bug: both heads using content_head instead of style_head/content_head
- Reduced lr 1e-3 → 3e-4 for stability
- Added NaN gradient detection + batch skip in train loop
- Clean 50-epoch training run completed (job 6588403, beta=1.0, lambda_sym=1.0, final avg loss: 13.0723)
- Fixed evaluate.py bugs (encoder/identity/pitch key errors), added run_eval.py runner
- Run 1 evaluation completed (job 6589342):
  - mig_pitch:    0.0093
  - mig_identity: 0.0808
  - dci_d:        0.0420
  - dci_c:        0.0248
  - dci_i:        0.9590
  - srr_db:       25.19 dB
- Results committed to git
- Resubmitted training with stronger hyperparameters (job 6589672, beta=4.0, lambda_sym=10.0)
## April 19, 2026
### Done

- Run 2 training completed (job 6589672, beta=4.0, lambda_sym=10.0, final avg loss: 19.6504)
- Run 2 evaluation completed (job 6595570):
  - mig_pitch:    0.0245  (↑ from 0.0093)
  - mig_identity: 0.0050  (↓ from 0.0808)
  - dci_d:        0.0456  (↑ from 0.0420)
  - dci_c:        0.0446  (↑ from 0.0248)
  - dci_i:        0.9590  (unchanged)
  - srr_db:       23.75   (↓ from 25.19)
- Key finding: stronger beta/lambda_sym improves pitch disentanglement but hurts
  identity separability — tradeoff worth discussing in writeup
- Eval outputs saved to logs/eval_run2_beta4_lsym10.out

## April 20, 2026
### Done
- Implemented β-VAE baseline (src/models/beta_vae.py, train_beta_vae.py)
- Run 3 training completed (job 6737470, beta=4.0, lambda_sym=50.0, 50 epochs, avg loss: 20.1638)
- β-VAE baseline training completed (job 6737469, beta=4.0, 50 epochs, avg loss: 19.6362)
- Fixed run_eval.py to auto-detect SymmetryVAE vs BetaVAE from checkpoint
- All evaluations completed:

| Metric       | β-VAE (β=4) | Run1 (β=1,λ=1) | Run2 (β=4,λ=10) | Run3 (β=4,λ=50) |
|--------------|-------------|-----------------|-----------------|-----------------|
| MIG pitch    | 0.0014      | 0.0093          | 0.0245          | 0.0109          |
| MIG identity | 0.0741      | 0.0808          | 0.0050          | 0.0016          |
| DCI-D        | 0.0466      | 0.0420          | 0.0456          | 0.0505          |
| DCI-C        | 0.0302      | 0.0248          | 0.0446          | 0.0422          |
| DCI-I        | 0.9590      | 0.9590          | 0.9590          | 0.9590          |
| SRR (dB)     | 23.52       | 25.19           | 23.75           | 23.62           |

- Key finding: symmetry constraint improves pitch disentanglement vs β-VAE baseline
  but creates tradeoff with identity separability at higher lambda_sym
- Submitting Run 4: beta=4.0, lambda_sym=10.0, 100 epochs for deeper training
- Implemented zero-shot evaluation pipeline (src/data/zeroshot_dataset.py, run_zeroshot_eval.py)
- Selected 8 holdout tracks across 8 genres as zero-shot test set:
  blues, bossa_nova, country, electronic, jazz, musical_theatre, reggae, world_folk
- Zero-shot eval completed for SymVAE Run2 and β-VAE:

| Song                          | β-VAE Style Var | SymVAE Style Var |
|-------------------------------|-----------------|------------------|
| Can't Play The Blues          | 0.2790          | 0.3255           |
| Dreaming Bout Being With You  | 0.2166          | 0.3079           |
| Nexus                         | 0.2447          | 0.2900           |
| Places                        | 0.2875          | 0.2462           |
| Sick Of Waiting               | 0.3016          | 0.2275           |
| Stolen Car                    | 0.4589          | 0.0977           |
| The Best In Me                | 0.3469          | 0.2020           |
| The Last To Know              | 0.2811          | 0.2519           |
| **MEAN**                      | **0.3020**      | **0.2436**       |

- Key finding: SymVAE produces more stable style encodings (lower Style Var)
  across out-of-distribution songs vs β-VAE baseline
- Stolen Car (electronic) shows strongest improvement (0.0977 vs 0.4589)

### Blocked
- Waiting for Run 4 (job 6744351) and Hierarchical VAE (job 6744408) to finish

### Up Next
- Evaluate Run 4 and Hierarchical VAE checkpoints
- Run zero-shot eval on Run 4 and Hierarchical VAE
- Write paper with team next weekend
- Deadline: April 29, 2026### Blocked
- Nothing currently blocked

### Up Next
- Wait for Run 4 (100 epochs) results
