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

### Up Next
- Wait for Run 2 training to complete (job 6589672)
- Run evaluation on Run 2 checkpoint
- Implement β-VAE baseline
- Write up results

