# Dataset Structure & HPC Path Configuration

## Your Actual HPC Layout (from May 7, 2026 extraction)

```
/scratch/mty236/ML_Project/data/

├── moisesdb_raw/
│   └── moisesdb/
│       └── moisesdb_v0.1/              ← IMPORTANT: nested moisesdb/
│           ├── <uuid-1>/
│           │   ├── bass/
│           │   │   ├── 0.wav
│           │   │   └── 1.wav
│           │   ├── drums/
│           │   ├── other/
│           │   └── vocals/
│           ├── <uuid-2>/
│           └── ... (240 tracks total)

└── nsynth_raw/
    └── nsynth-train/
        ├── examples.json              ← Metadata for ~289k examples
        ├── audio/
        │   ├── bass_acoustic_000.wav  ← Filter: bass_*
        │   ├── bass_acoustic_001.wav
        │   ├── bass_electronic_000.wav
        │   ├── brass_acoustic_000.wav ← Ignored (non-bass)
        │   └── ... (~289k files, ~60-65k bass)
        └── ... (other split structure files)

Conda env: /scratch/mty236/conda_envs/musicvae
Repo:      /scratch/mty236/ML_Project/Disentangled-Representations-Learning-for-Music-Processing/
```

---

## Compatibility Check with CQT Pipeline

### ✅ MoisesDB Loader Compatibility

**Our loader expects:**
```python
MoisesDBBass(root="/path/to/dataset_root")
# Looks for: <root>/<track_id>/bass/*.wav
```

**Your structure:**
```
/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1/
├── <uuid-1>/bass/*.wav
├── <uuid-2>/bass/*.wav
└── ... (240 tracks)
```

**Result:** ✅ **Fully Compatible**

Set environment variable:
```bash
export MOISESDB_ROOT=/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1
```

The loader will:
1. Discover 240 track directories
2. Find bass/ subfolder in each
3. Filter to tracks with bass stems
4. Load ~237 bass stems (as per Adarsh's PROGRESS.md)

---

### ✅ NSynth Loader Compatibility

**Our loader expects:**
```python
NSynthBass(root="/path/to/nsynth_root", split="train")
# Looks for: <root>/nsynth-train/examples.json
#            <root>/nsynth-train/audio/bass_*.wav
```

**Your structure:**
```
/scratch/mty236/ML_Project/data/nsynth_raw/nsynth-train/
├── examples.json          (metadata for all 289k examples)
└── audio/
    ├── bass_acoustic_*.wav (~60-65k bass examples)
    ├── bass_electronic_*.wav
    └── other_*.wav        (ignored by NSynthBass filter)
```

**Result:** ✅ **Fully Compatible**

Set environment variable:
```bash
export NSYNTH_ROOT=/scratch/mty236/ML_Project/data/nsynth_raw
```

The loader will:
1. Open `nsynth-train/examples.json`
2. Filter to rows where `instrument_family_str == "bass"`
3. Load from `nsynth-train/audio/bass_*.wav`
4. Get ~60-65k bass examples as expected

---

## ⚠️ Important Notes (from your context)

### 1. NSynth Train-Only (No Valid/Test)

You only downloaded **nsynth-train** (~289k examples, 22.2 GB).

**Implication for evaluation:**
- ✅ Can train on NSynth bass
- ❌ Cannot use NSynth Valid/Test for OOD evaluation (would need separate download)
- ❌ All MIG/DCI metrics use Train split for identity check → **not clean OOD probe**

**If you want OOD NSynth evaluation later:**
```bash
# Download valid/test splits (~330 MB total, fast)
wget -c http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
wget -c http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
tar -xzf nsynth-valid.jsonwav.tar.gz -C /scratch/mty236/ML_Project/data/nsynth_raw/
tar -xzf nsynth-test.jsonwav.tar.gz -C /scratch/mty236/ML_Project/data/nsynth_raw/
```

For now: **skip this** (not in 2-day timeline).

### 2. MoisesDB Zero-Shot Leakage

You noted 8 MoisesDB training tracks were in zero-shot dataset. This is **separate from current training** (you're not using zero-shot in 2-day run), but document it for reproducibility.

### 3. Nesting Level (moisesdb/moisesdb_v0.1/)

**Critical:** The extra nesting from your v3 unzip script means:

```bash
# WRONG:
export MOISESDB_ROOT=/scratch/mty236/ML_Project/data/moisesdb_raw/

# CORRECT:
export MOISESDB_ROOT=/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1

# Verify:
ls $MOISESDB_ROOT | head
# Should show UUIDs like: a1b2c3d4-e5f6-..., 0a1b2c3d-..., etc.
# NOT: moisesdb, moisesdb_v0.1, etc.
```

---

## HPC Environment Setup (2-Day Training)

### Session Start Script

Create `/scratch/mty236/setup_env.sh`:

```bash
#!/bin/bash

# Activate conda env
source /scratch/mty236/conda_envs/musicvae/bin/activate

# Set dataset paths (critical for compatibility)
export NSYNTH_ROOT=/scratch/mty236/ML_Project/data/nsynth_raw
export MOISESDB_ROOT=/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1

# Set cache output directory
export NSYNTH_CACHE=/scratch/mty236/ML_Project/caches/nsynth_cqt
mkdir -p $NSYNTH_CACHE

# Navigate to repo
cd /scratch/mty236/ML_Project/Disentangled-Representations-Learning-for-Music-Processing

# Verify paths exist
echo "=== Dataset Paths ==="
echo "NSYNTH_ROOT: $NSYNTH_ROOT"
echo "  Exists: $(test -d $NSYNTH_ROOT && echo YES || echo NO)"
echo "  Contents: $(ls -d $NSYNTH_ROOT/nsynth-* 2>/dev/null | wc -l) split dirs"

echo ""
echo "MOISESDB_ROOT: $MOISESDB_ROOT"
echo "  Exists: $(test -d $MOISESDB_ROOT && echo YES || echo NO)"
echo "  Contents: $(find $MOISESDB_ROOT -maxdepth 1 -type d | wc -l) track folders"

echo ""
echo "NSYNTH_CACHE: $NSYNTH_CACHE"
echo "  Exists: $(test -d $NSYNTH_CACHE && echo YES || echo NO)"

echo ""
echo "=== Ready for training ==="
```

**Usage:**
```bash
source /scratch/mty236/setup_env.sh
# Prints verification, sets all env vars
```

---

## Day 1: Verification Commands

### 1. Check NSynth Structure

```bash
source /scratch/mty236/setup_env.sh

# Verify examples.json exists
test -f $NSYNTH_ROOT/nsynth-train/examples.json && echo "✅ examples.json found" || echo "❌ missing"

# Count bass examples
python -c "
import json
with open('$NSYNTH_ROOT/nsynth-train/examples.json') as f:
    data = json.load(f)
bass_count = sum(1 for v in data.values() if v.get('instrument_family_str') == 'bass')
print(f'NSynth train bass count: {bass_count}')
print(f'NSynth train total: {len(data)}')
"

# Expected output:
# NSynth train bass count: 60000-65000
# NSynth train total: 289205
```

### 2. Check MoisesDB Structure

```bash
source /scratch/mty236/setup_env.sh

# Count track folders
TRACK_COUNT=$(find $MOISESDB_ROOT -maxdepth 1 -type d -not -name 'moisesdb*' | wc -l)
echo "MoisesDB track folders: $TRACK_COUNT (expected ~240)"

# Count bass stems
BASS_STEMS=$(find $MOISESDB_ROOT -type d -name "bass" | wc -l)
echo "MoisesDB tracks with bass: $BASS_STEMS"

# Sample one track
SAMPLE=$(find $MOISESDB_ROOT -maxdepth 1 -type d -not -name 'moisesdb*' | head -1)
echo "Sample track: $SAMPLE"
echo "  Contents: $(ls -1 $SAMPLE)"
```

### 3. Smoke Test

```bash
source /scratch/mty236/setup_env.sh

# Quick synthetic test (no dataset I/O)
python -m src.training.cli +trainer.fast_dev_run=true

# Expected: completes in <2 min with no errors
```

---

## Day 1 Evening: Cache Pre-computation

### SLURM Job Script

Create `/scratch/mty236/job_cache_nsynth.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=nsynth_cqt_cache
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/scratch/mty236/logs/cache_%j.out
#SBATCH --error=/scratch/mty236/logs/cache_%j.err

source /scratch/mty236/setup_env.sh

mkdir -p /scratch/mty236/logs

# Cache train split (should complete in ~20-30 min on A100)
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda

echo "Train cache complete: $NSYNTH_CACHE/train_cqt.h5"
ls -lh $NSYNTH_CACHE/train_cqt.h5
```

**Submit:**
```bash
mkdir -p /scratch/mty236/logs
sbatch /scratch/mty236/job_cache_nsynth.slurm

# Monitor
squeue -u mty236
tail -f /scratch/mty236/logs/cache_*.out
```

---

## Day 2: Training

### Training Job Script (NSynth)

Create `/scratch/mty236/job_train_nsynth.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sc_vae_nsynth
#SBATCH --time=06:00:00
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --output=/scratch/mty236/logs/train_nsynth_%j.out
#SBATCH --error=/scratch/mty236/logs/train_nsynth_%j.err

source /scratch/mty236/setup_env.sh

# Set cache paths for training
export NSYNTH_CACHE_TRAIN=$NSYNTH_CACHE/train_cqt.h5
export NSYNTH_CACHE_VALID=$NSYNTH_CACHE/valid_cqt.h5  # Needed if valid cache exists

mkdir -p /scratch/mty236/logs

# NSynth training (50 epochs, ~4-6 hrs on 2× GPU)
python -m src.training.cli \
    data=nsynth_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=0,1 \
    trainer.strategy=ddp_find_unused_parameters_false \
    lit.beta_s=1.0 \
    lit.beta_c=1.0 \
    lit.lambda_inv=1.0 \
    lit.lambda_equi=1.0 \
    seed=0

echo "Training complete"
```

**Submit:**
```bash
sbatch /scratch/mty236/job_train_nsynth.slurm
squeue -u mty236
tail -f /scratch/mty236/logs/train_nsynth_*.out
```

### Training Job Script (MoisesDB)

Create `/scratch/mty236/job_train_moisesdb.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sc_vae_moisesdb
#SBATCH --time=06:00:00
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --output=/scratch/mty236/logs/train_moisesdb_%j.out
#SBATCH --error=/scratch/mty236/logs/train_moisesdb_%j.err

source /scratch/mty236/setup_env.sh

mkdir -p /scratch/mty236/logs

# MoisesDB training (50 epochs, ~3-5 hrs on 2× GPU, on-the-fly CQT)
python -m src.training.cli \
    data=moisesdb_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=0,1 \
    trainer.strategy=ddp_find_unused_parameters_false \
    lit.beta_s=1.0 \
    lit.beta_c=1.0 \
    lit.lambda_inv=1.0 \
    lit.lambda_equi=1.0 \
    seed=0

echo "Training complete"
```

**Submit (after NSynth completes or on different GPUs):**
```bash
sbatch /scratch/mty236/job_train_moisesdb.slurm
```

---

## Output Directory Structure

After all runs:

```
/scratch/mty236/ML_Project/

├── caches/
│   └── nsynth_cqt/
│       ├── train_cqt.h5        (1.3 GB)
│       ├── valid_cqt.h5        (150 MB)  [if you pre-compute]
│       ├── train.log
│       └── valid.log

├── logs/
│   ├── cache_12345.out         [cache job logs]
│   ├── train_nsynth_12346.out  [NSynth training logs]
│   └── train_moisesdb_12347.out [MoisesDB training logs]

├── Disentangled-Representations-Learning-for-Music-Processing/
│   └── Lightning_logs/
│       ├── version_0/          [NSynth run]
│       │   ├── checkpoints/
│       │   └── events.out.tfevents
│       └── version_1/          [MoisesDB run]
│           ├── checkpoints/
│           └── events.out.tfevents

└── data/
    ├── moisesdb_raw/           [unchanged]
    └── nsynth_raw/             [unchanged]
```

---

## Troubleshooting: Path Issues

### Error: `FileNotFoundError: missing manifest`

**Cause:** Wrong NSYNTH_ROOT

**Fix:**
```bash
# Check what you set
echo $NSYNTH_ROOT

# Verify structure
ls -la $NSYNTH_ROOT/nsynth-train/examples.json
# Should exist and be ~10 MB

# If not found, set correctly
export NSYNTH_ROOT=/scratch/mty236/ML_Project/data/nsynth_raw
```

### Error: `no MoisesDB tracks with bass stems under $MOISESDB_ROOT`

**Cause:** Wrong MOISESDB_ROOT (likely pointing at parent dir instead of moisesdb_v0.1)

**Fix:**
```bash
# Check what you set
echo $MOISESDB_ROOT

# Verify structure
ls $MOISESDB_ROOT | head
# Should show UUIDs like: a1b2c3d4-e5f6-..., not "moisesdb" or "moisesdb_v0.1"

# If wrong, set correctly
export MOISESDB_ROOT=/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1
```

### Error: Cache HDF5 not found during training

**Cause:** NSYNTH_CACHE_TRAIN not set or wrong path

**Fix:**
```bash
# Make sure setup_env.sh was sourced
source /scratch/mty236/setup_env.sh

# Verify cache was created
ls -lh $NSYNTH_CACHE/*.h5

# If missing, re-run cache script
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda
```

---

## Summary: Your Datasets Are Compatible ✅

| Dataset | Current Path | Loader Expects | Status |
|---------|---|---|---|
| **NSynth** | `/scratch/mty236/ML_Project/data/nsynth_raw` | `<root>/nsynth-train/` | ✅ Match |
| **MoisesDB** | `/scratch/mty236/ML_Project/data/moisesdb_raw/moisesdb/moisesdb_v0.1` | `<root>/<uuid>/bass/` | ✅ Match |

**No re-extraction or re-downloading needed!**

Just set env vars correctly and start training.

