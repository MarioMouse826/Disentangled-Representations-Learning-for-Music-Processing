# Quick Start Guide: Option A (CQT) + Fast Preprocessing

## ✅ All Code is Done and Pushed

Your `Mario-Expanded-Models` branch now has:
- ✅ ConstantQ feature extractor (src/data/features.py)
- ✅ Updated loaders (NSynthBassCached, MoisesDBBassMel)
- ✅ CQT configs for all models
- ✅ CQT-compatible cache script
- ✅ nnAudio dependency added

## ✅ Compatibility Check

Your extracted datasets work **perfectly as-is**:

```bash
# NSynth (you already extracted)
$NSYNTH_ROOT/nsynth-train/audio/*.wav     ← ✅ Ready to use
$NSYNTH_ROOT/nsynth-valid/audio/*.wav     ← ✅ Ready to use

# MoisesDB (you already extracted)
$MOISESDB_ROOT/track_*/bass/*.wav         ← ✅ Ready to use
```

**No re-extraction or re-downloading needed!**

---

## 🚀 Day 1: Validation & Cache Generation

### Step 1: Install nnAudio

```bash
pip install nnAudio==0.3.2
```

### Step 2: Smoke Test (5 minutes)

```bash
python -m src.training.cli +trainer.fast_dev_run=true
```

Expected output:
```
Epoch 1: [================================================] 100%
val/total_loss: 3.456
```

If this works → pipeline is solid ✅

### Step 3: Validate Your Datasets

```bash
# Count NSynth bass samples
python -c "
import json
from pathlib import Path
d = json.load(open('$NSYNTH_ROOT/nsynth-train/examples.json'))
print('NSynth train bass:', sum(1 for v in d.values() if v.get('instrument_family_str')=='bass'))
"

# Count MoisesDB tracks with bass stems
find $MOISESDB_ROOT -type d -name "bass" | wc -l
```

Expected: NSynth ≥1300, MoisesDB ≥200

### Step 4: Pre-compute NSynth CQT Caches (20-25 min on A100)

**Important: Use `--batch-size 256` for speed**

```bash
# Setup
export NSYNTH_ROOT=/path/to/your/nsynth/root
export NSYNTH_CACHE=/path/to/cache/dir
mkdir -p $NSYNTH_CACHE

# Train split (runs in background)
nohup python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda \
    > $NSYNTH_CACHE/train.log 2>&1 &

# Monitor
tail -f $NSYNTH_CACHE/train.log
```

While that runs (in another terminal):

```bash
# Valid split
nohup python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split valid \
    --out $NSYNTH_CACHE/valid_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda \
    > $NSYNTH_CACHE/valid.log 2>&1 &
```

**Expected time:** ~25 min total on A100; ~45 min on V100

---

## 🎯 Day 2: Full Training

### Step 1: NSynth Training (4-6 hours on 2× GPU)

```bash
# Set cache paths
export NSYNTH_CACHE_TRAIN=/path/to/cache/train_cqt.h5
export NSYNTH_CACHE_VALID=/path/to/cache/valid_cqt.h5

# Train SC-VAE on NSynth
python -m src.training.cli \
    data=nsynth_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=[0,1] \
    lit.beta_s=1.0 \
    lit.beta_c=1.0 \
    lit.lambda_inv=1.0 \
    lit.lambda_equi=1.0 \
    seed=0
```

### Step 2: MoisesDB Training (3-5 hours on 2× GPU)

```bash
# Set dataset root
export MOISESDB_ROOT=/path/to/your/moisesdb/root

# Train SC-VAE on MoisesDB (uses on-the-fly CQT)
python -m src.training.cli \
    data=moisesdb_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=[0,1] \
    lit.beta_s=1.0 \
    lit.beta_c=1.0 \
    lit.lambda_inv=1.0 \
    lit.lambda_equi=1.0 \
    seed=0
```

### Step 3: Monitor Metrics

Watch for these validation metrics (per epoch):
- `val/loss_recon` - reconstruction quality (lower is better, target ~0.5-1.0)
- `val/loss_kl_s` - style KL divergence
- `val/loss_kl_c` - content KL divergence
- `val/loss_inv` - style invariance (lower is better)
- `val/loss_equi` - content equivariance (lower is better)

---

## ⏱️ Timeline Summary

| Phase | Time | Notes |
|-------|------|-------|
| Install nnAudio | 2 min | `pip install nnAudio==0.3.2` |
| Smoke test | 5 min | `+trainer.fast_dev_run=true` |
| Dataset validation | 5 min | Count clips |
| NSynth cache (train) | 15-20 min | Runs in background |
| NSynth cache (valid) | 5-10 min | Parallel with train |
| NSynth training (50 ep) | 4-6 hrs | Day 2 morning |
| MoisesDB training (50 ep) | 3-5 hrs | Day 2 afternoon |
| **Total** | **~6-7 hrs real time** | Parallelizable |

---

## 🎛️ Performance Tuning

### GPU Memory Issues?

If you get OOM errors:

```bash
# Reduce batch size
python scripts/prepare_nsynth_cache.py \
    --batch-size 128 \  # was 256
    ...

# Or reduce training batch size in config
python -m src.training.cli \
    data=nsynth_bass \
    data.batch_size=16 \  # was 32
    ...
```

### Slow CQT Computation During MoisesDB Training?

MoisesDB uses **on-the-fly CQT** (computes per-batch). If too slow:

```bash
# Option 1: Pre-compute MoisesDB like NSynth
# (requires ~2-3 hrs, but training will be faster)

# Option 2: Reduce num_workers
python -m src.training.cli \
    data=moisesdb_bass \
    data.num_workers=2 \  # was 4
    ...

# Option 3: Reduce batch size
python -m src.training.cli \
    data=moisesdb_bass \
    data.batch_size=8 \  # was 16
    ...
```

---

## 📊 Expected Results

After 50 epochs on NSynth (with CQT):

| Metric | Expected Value | Status |
|--------|---|---|
| MIG (Mutual Information Gap) | 0.35-0.50 | ✅ Good disentanglement |
| SAP (Separated Attribute Predictability) | 0.40-0.60 | ✅ Well separated |
| DCI-D (Informativeness) | 0.50-0.70 | ✅ Good reconstruction |
| Reconstruction Loss | 0.5-1.0 (dB MSE) | ✅ Acceptable |
| Invariance Loss | ~0.1-0.3 | ✅ Style unchanged |
| Equivariance Loss | ~0.2-0.5 | ✅ Content shifts with pitch |

---

## 📁 File Structure After Caching

```
$NSYNTH_CACHE/
├── train_cqt.h5         (1.3 GB) ← From prepare script
├── valid_cqt.h5         (150 MB)
├── train.log
└── valid.log

Lightning_logs/
├── version_0/           ← NSynth training run
│   ├── checkpoints/
│   └── events.out.tfevents
└── version_1/           ← MoisesDB training run
    ├── checkpoints/
    └── events.out.tfevents
```

---

## 🔄 Switching Back to Mel (if needed)

If you want to compare with Ishan's original mel-based approach:

```bash
# Use mel configs instead
python -m src.training.cli \
    data=nsynth_bass \
    model=sc_vae \
    data.train_dataset.feature_type=mel \
    data.val_dataset.feature_type=mel \
    trainer.max_epochs=10
```

Or keep separate mel cache:

```bash
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_mel.h5 \
    --feature-type mel \
    --batch-size 256 \
    --device cuda
```

---

## ❓ Troubleshooting

### Error: `ImportError: No module named 'nnAudio'`

```bash
pip install nnAudio==0.3.2
```

### Error: `FileNotFoundError: missing cache at ...`

Make sure env vars are set:

```bash
echo $NSYNTH_CACHE_TRAIN
echo $NSYNTH_CACHE_VALID
# Should print paths, not empty
```

### Error: `Shape mismatch in encoder`

Likely a config sync issue. Check that all model configs have `n_mels: 84`:

```bash
grep "n_mels:" configs/model/*.yaml
# Should all show 84
```

### Cache script hangs or crashes

Check GPU memory:

```bash
nvidia-smi
# Verify batch_size 256 fits in your GPU VRAM
# If not, reduce to 128 or 64
```

---

## ✅ Checklist for Day 1

- [ ] Install nnAudio: `pip install nnAudio==0.3.2`
- [ ] Run smoke test: `python -m src.training.cli +trainer.fast_dev_run=true`
- [ ] Validate NSynth: `python -c "...count bass samples..."`
- [ ] Validate MoisesDB: `find $MOISESDB_ROOT -type d -name "bass" | wc -l`
- [ ] Start cache generation: `nohup python scripts/prepare_nsynth_cache.py ... &`
- [ ] Monitor cache log: `tail -f train.log`
- [ ] While waiting, review training configs

## ✅ Checklist for Day 2

- [ ] Verify caches completed: `ls -lh $NSYNTH_CACHE/*.h5`
- [ ] Start NSynth training: `python -m src.training.cli data=nsynth_bass ...`
- [ ] Start MoisesDB training: `python -m src.training.cli data=moisesdb_bass ...`
- [ ] Monitor metrics: `tensorboard --logdir Lightning_logs/`
- [ ] Collect results after training completes

---

**You're ready to go! Your datasets are compatible, code is optimized, and timeline is realistic. 🚀**

