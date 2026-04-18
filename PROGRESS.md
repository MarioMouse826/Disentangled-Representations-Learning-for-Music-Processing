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
### Blocked
- Full Magenta import broken — Keras 3 / RNNCell incompatibility
- Datasets not downloaded yet — need HPC for full run

### Up Next
- Download NSynth + MoisesDB on HPC
- Set up Greene environment
- Run full training pipeline on HPC
