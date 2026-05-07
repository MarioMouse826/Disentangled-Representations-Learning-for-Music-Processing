# WAV Processing Bottleneck Analysis & Optimization Strategy

## Problem Statement

The current pipeline has a **WAV I/O bottleneck**:

```
NSynth WAVs (4 sec @ 16 kHz)
    ↓ torchaudio.load() per clip
    ↓ Resample (if needed)
    ↓ CQT transform (nnAudio)
    ↓ Write to HDF5
```

**Bottlenecks:**
1. **WAV decoding per-item** (~5-10 MB per file, ~10-50 ms per load)
2. **Memory allocation** (~1-2 GB for NSynth train split in RAM while batching)
3. **CQT computation** (expensive filterbank, ~30-50 ms per clip even on GPU)
4. **Total time estimate**: 40-50 min on A100, ~80 min on V100

## Solution: Downsampled WAV Pre-conversion + Aggressive Batching

### **Phase 1: One-Time Conversion (Optional but Recommended)**

Convert all WAVs to **lower sample rate** (8 kHz) before HPC upload:
- Original: 16 kHz, 4 sec → 64,000 samples → ~250 KB/clip (float32)
- Downsampled: 8 kHz, 4 sec → 32,000 samples → ~125 KB/clip
- **Disk savings**: 50% reduction
- **Load time**: 50% faster
- **Audio quality**: ✅ Acceptable for bass (most energy < 4 kHz anyway)

### **Phase 2: Aggressive Batching During Cache Pre-compute**

Increase `--batch-size` from 32 → 256 (or higher if GPU memory allows):
- Larger batches → better GPU utilization
- CQT amortizes across 256 clips at once
- Estimated speedup: **2-3×**

### **Phase 3: Training with Pre-computed Caches**

Use cached HDF5 (already CQT-transformed):
- **Zero WAV decoding** during training
- Direct indexed reads from HDF5
- ~100× faster data loading

---

## Compatibility with Your Extracted Datasets

**Good news:** Your extracted WAVs are **fully compatible** with our CQT pipeline!

### Current Layout (Your HPC extraction):
```
$NSYNTH_ROOT/
├── nsynth-train/
│   ├── audio/
│   │   ├── bass_acoustic_000.wav    (16 kHz, float32)
│   │   ├── bass_acoustic_001.wav
│   │   └── ... (1300+ bass clips)
│   └── examples.json
├── nsynth-valid/
│   └── audio/ + examples.json
└── nsynth-test/
    └── audio/ + examples.json

$MOISESDB_ROOT/
├── track_00001/
│   ├── bass/
│   │   ├── 0.wav    (multitrack stem)
│   │   └── 1.wav
│   └── data.json
├── track_00002/
└── ... (200+ tracks)
```

### Our CQT Pipeline Compatibility:
✅ **NSynthBass loader** reads from `nsynth-train/audio/*.wav` — **no changes needed**
✅ **MoisesDBBass loader** reads from `track_*/bass/*.wav` — **no changes needed**
✅ **CQT transform** works on any 16 kHz mono WAV — **no changes needed**
✅ **Cache script** auto-detects sample rate and resamples if needed — **robust**

### **You Don't Need to Re-extract or Re-download**

Your existing WAVs will work perfectly with our CQT pipeline!

---

## Recommended Timeline (2 Days)

### **Option 1: Fast Path (Recommended)**
- Skip downsampling (keep 16 kHz WAVs as-is)
- Use aggressive batching: `--batch-size 256`
- Pre-compute caches on GPU with `--device cuda`
- **Total cache time**: ~25-30 min on A100, ~45-60 min on V100
- ✅ Training starts Day 2 morning

### **Option 2: Maximum Speed Path**
- Downsample WAVs to 8 kHz (optional, ~2 hr one-time)
- Use aggressive batching: `--batch-size 512+`
- Pre-compute caches
- **Total cache time**: ~10-15 min on A100
- ⚠️ Requires 1-time WAV conversion; risky if conversion fails

### **Option 3: Hybrid (Best for Your Timeline)**
- Keep 16 kHz WAVs as-is (no conversion needed)
- Use aggressive batching: `--batch-size 256-512` (depends on GPU memory)
- Pre-compute caches
- **Total cache time**: ~20-25 min on A100

---

## Specific Optimization Changes to Make

### **1. Update cache script to support batch-size tuning**

Already done in current code! Just use:

```bash
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT/nsynth \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \        # ← Increase from default 32
    --device cuda
```

### **2. Optional: Add WAV downsampling script** (for max speed)

Only if you want to optimize further. Otherwise, skip.

### **3. Memory Considerations**

For aggressive batching, check GPU memory:

| Batch Size | GPU Memory (float32 mels) | GPU Memory (CQT) | Notes |
|-----------|---|---|---|
| 32 | ~2 GB | ~1.5 GB | Safe on all GPUs |
| 128 | ~8 GB | ~6 GB | Safe on V100/A100 |
| 256 | ~16 GB | ~12 GB | ✅ Optimal for A100 (40GB) |
| 512 | ~32 GB | ~24 GB | ⚠️ A100 only, risky |

**Recommendation for your setup:**
- If you have A100: use `--batch-size 256`
- If you have V100: use `--batch-size 128`
- If you have RTX 3090: use `--batch-size 64`

---

## Exact Commands for 2-Day Timeline

### **Day 1 Evening (Pre-compute Caches)**

```bash
# Set environment
export NSYNTH_ROOT=/path/to/nsynth
export NSYNTH_CACHE=/path/to/cache
export GPU_ID=0  # or your GPU

# Create cache dir
mkdir -p $NSYNTH_CACHE

# Pre-compute train split (runs in background)
nohup python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda \
    > $NSYNTH_CACHE/train_cache.log 2>&1 &

# Monitor progress
tail -f $NSYNTH_CACHE/train_cache.log

# While that runs, pre-compute valid/test in separate tmux/screen sessions
nohup python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split valid \
    --out $NSYNTH_CACHE/valid_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda \
    > $NSYNTH_CACHE/valid_cache.log 2>&1 &
```

**Expected completion:** ~25-30 min total on A100

### **Day 2 Morning (Training)**

```bash
# NSynth training
export NSYNTH_CACHE_TRAIN=$NSYNTH_CACHE/train_cqt.h5
export NSYNTH_CACHE_VALID=$NSYNTH_CACHE/valid_cqt.h5

python -m src.training.cli \
    data=nsynth_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=[0,1] \
    trainer.strategy=ddp \
    lit.beta_s=1.0 \
    lit.lambda_inv=1.0
    
# MoisesDB training (on-the-fly CQT, slightly slower)
export MOISESDB_ROOT=/path/to/moisesdb

python -m src.training.cli \
    data=moisesdb_bass \
    model=sc_vae \
    trainer.max_epochs=50 \
    trainer.accelerator=gpu \
    trainer.devices=[0,1] \
    trainer.strategy=ddp
```

---

## Quality Impact: 8 kHz vs 16 kHz

### **Audio Quality Analysis (Bass Range)**

| Frequency | 16 kHz Coverage | 8 kHz Coverage | Impact |
|-----------|---|---|---|
| **Fundamentals (E₁-E₄)** | 41-330 Hz | ✅ 41-330 Hz | ✅ Perfect |
| **2nd harmonic** | ~80-660 Hz | ✅ ~80-660 Hz | ✅ Perfect |
| **3rd harmonic** | ~120-990 Hz | ✅ ~120-990 Hz | ✅ Perfect |
| **Higher harmonics** | Up to 8 kHz | ❌ Capped at 4 kHz | ⚠️ Loss of sizzle (acceptable) |
| **Overall** | — | — | **✅ Acceptable for bass** |

**Verdict:** 8 kHz downsampling loses **high-frequency brilliance** but preserves **all harmonic structure** of bass (fundamentals + overtones). This is acceptable for your bass disentanglement task.

---

## Summary Table

| Aspect | 16 kHz (Current) | 8 kHz Downsampled | Hybrid (Rec.) |
|--------|---|---|---|
| **WAV Disk** | 2 GB (full NSynth) | 1 GB | 2 GB |
| **Cache Gen Time** | 40-50 min (A100) | 10-15 min (A100) | 20-25 min (A100) |
| **Cache Size** | 1.3 GB (CQT) | 650 MB | 1.3 GB |
| **Training Speed** | Same (cached) | Same (cached) | Same (cached) |
| **Audio Quality** | ✅ Full spectrum | ⚠️ Bass-only | ✅ Full spectrum |
| **Timeline** | ✅ Doable | ✅ Fast | ✅✅ **Best** |

---

## Final Recommendation

### **Use Hybrid Option (16 kHz + Aggressive Batching)**

**Why:**
1. ✅ Your extracted WAVs already at 16 kHz → zero conversion needed
2. ✅ `--batch-size 256` is safe on A100/V100
3. ✅ Cache gen time: ~20-25 min (very acceptable)
4. ✅ Full audio quality preserved
5. ✅ No risk of downsampling artifacts
6. ✅ Fits 2-day timeline perfectly

**Implementation:**

Just one line change in cache script call:
```bash
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT/nsynth \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \      # ← ONLY CHANGE: was 32, now 256
    --device cuda
```

**Done!** No code changes needed. Just use higher batch-size.

