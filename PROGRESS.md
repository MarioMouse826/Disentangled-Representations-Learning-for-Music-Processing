# Adarsh — Progress Log

## Branch: adarsh/data-pipeline

---

## April 17, 2026

### Done
- Cloned MusicVAE (Magenta) repo
- Set up musicvae conda env (Python 3.9)
- Installed core dependencies: tensorflow, librosa, numpy, note-seq, torch, torchaudio
- Cloned team repo, created and pushed adarsh/data-pipeline branch
- Set up project folder structure (data/, src/, notebooks/, results/)

### Blocked
- Full Magenta import broken — Keras 3 / RNNCell incompatibility
- Need to ask Mario: pretrained MusicVAE checkpoint or train from scratch?

### Up Next
- NSynth dataset class (src/data/nsynth_dataset.py)
- Mel spectrogram + pitch shift pipeline
- Download NSynth bass subset on HPC

---
## Log format: add a new date block each session