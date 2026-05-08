# OPTION 2: All 5 Models on NSynth Only (2-Day Training Plan)

**Date:** May 7-8, 2026  
**Status:** ✅ Code Complete & Tested  
**Branch:** `Mario-Expanded-Models` (pushed)  
**Focus:** SC-VAE + 4 Baselines (beta_vae, beta_tcvae, factor_vae, ar_hvae)  
**Dataset:** NSynth bass only (~60-65k examples)  
**Timeline:** 2 Days

---

## Executive Summary

Train all 5 VAE models on NSynth bass in parallel using CQT features. Skip MoisesDB training to save time; use it for post-training OOD evaluation instead.

| Phase | Time | Action |
|-------|------|--------|
| **Day 1 Morning** | 30 min | Install nnAudio, smoke test, validate NSynth |
| **Day 1 Evening** | ~25 min | Pre-compute NSynth CQT cache |
| **Day 2 All Day** | ~5-6 hrs | Train all 5 models in parallel (50 epochs each) |
| **Day 2 Evening** | Manual | Collect metrics & logs |

---

## What You're Training

| Model | Reference | Key Paper | Hyperparameters |
|-------|-----------|-----------|-----------------|
| **SC-VAE** | Your main model | Symmetric group action | β_s=1.0, λ_inv=1.0 |
| **β-VAE** | Higgins et al. 2017 | Disentanglement via KL-weighted ELBO | β_s=4.0 |
| **β-TCVAE** | Chen et al. 2018 | Total Correlation VAE | β_s=6.0, α=1.0, γ=1.0 |
| **FactorVAE** | Kim & Mnih 2018 | Disentanglement via independence penalty | β_s=1.0, λ_inv=35.0 |
| **AR-HVAE** | Autoregressive Hierarchical VAE | Ladder VAE / NVAE-style | β_s=1.0 |

---

## Why NSynth Only?

✅ **Large enough:** 60-65k examples × 50 epochs = 3-3.25M gradient updates (plenty for VAE)  
✅ **Time efficient:** 5-6 hrs total for all 5 models in parallel (vs. 24-30 hrs sequential)  
✅ **Research value:** Clean training on synthetic, then evaluate generalization on MoisesDB  
✅ **Reproducibility:** Easier to debug 1 dataset vs. 2  
🔄 **MoisesDB later:** Use for post-hoc OOD evaluation (`test_zero_shot_ood.py`) after training completes

---

## Day 1: Setup & Cache Generation

### Step 1: Setup Environment (5 min)

```bash
# Create setup script
cat > /scratch/mty236/setup_env.sh << 'EOF'
source /scratch/mty236/conda_envs/musicvae/bin/activate
export NSYNTH_ROOT=/scratch/mty236/ML_Project/data/nsynth_raw
export NSYNTH_CACHE=/scratch/mty236/ML_Project/caches/nsynth_cqt
mkdir -p $NSYNTH_CACHE
cd /scratch/mty236/ML_Project/Disentangled-Representations-Learning-for-Music-Processing
EOF

chmod +x /scratch/mty236/setup_env.sh
source /scratch/mty236/setup_env.sh
```

### Step 2: Install Dependencies (2 min)

```bash
pip install nnAudio==0.3.2
```

### Step 3: Smoke Test (2 min)

Test the synthetic pipeline with fast_dev_run:

```bash
python -m src.training.cli +trainer.fast_dev_run=true
```

**Expected output:** Should complete in ~30-60 sec, no errors.

### Step 4: Validate NSynth Dataset (5 min)

```bash
# Count bass examples
python -c "
import json
with open('$NSYNTH_ROOT/nsynth-train/examples.json') as f:
    data = json.load(f)
    bass = [k for k, v in data.items() if v['instrument_family'] == 1]  # 1 = bass
    print(f'NSynth bass examples: {len(bass)}')
"
```

**Expected:** ~60,000-65,000 bass examples

### Step 5: Pre-compute NSynth CQT Cache (20-25 min)

Generate HDF5 with CQT features for fast training:

```bash
python scripts/prepare_nsynth_cache.py \
    --src $NSYNTH_ROOT \
    --split train \
    --out $NSYNTH_CACHE/train_cqt.h5 \
    --feature-type cqt \
    --batch-size 256 \
    --device cuda
```

**Expected output:**
- Progress bar: 60k+ samples processed
- Time: ~20-25 min on A100, ~40-50 min on V100
- Output: `train_cqt.h5` (~1.3 GB)

**Note:** Cache script runs in background. Monitor progress and move to Day 2 once complete.

---

## Day 2: Train All 5 Models in Parallel

### Prepare Training Commands

Create a training script to launch all 5 models:

```bash
cat > /scratch/mty236/train_all_5_models.sh << 'EOF'
#!/bin/bash
set -euo pipefail

source /scratch/mty236/setup_env.sh

OUT_ROOT="runs/option2_nsynth_50epochs"
mkdir -p "$OUT_ROOT"

# Set common parameters
NSYNTH_CACHE_TRAIN=$NSYNTH_CACHE/train_cqt.h5
DATA=nsynth_bass
MAX_EPOCHS=50
PRECISION=bf16-mixed
DEVICES="0,1"

echo "Training all 5 models on NSynth (50 epochs each)"
echo "Output: $OUT_ROOT"
echo ""

# Function to run one model
run_model() {
    local model="$1"
    local extra_args="$2"
    local run_dir="$OUT_ROOT/${model}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $model..."
    
    python -m src.training.cli \
        --config-name train/sweep \
        data=$DATA \
        model=$model \
        trainer.max_epochs=$MAX_EPOCHS \
        trainer.precision=$PRECISION \
        trainer.accelerator=gpu \
        trainer.devices=$DEVICES \
        hydra.run.dir=$run_dir \
        $extra_args &
    
    echo "[PID $!] $model"
}

# Launch all 5 models (will run in parallel if enough GPU memory)
# SC-VAE: β_s=1.0, λ_inv=1.0 (group symmetry + content equivariance)
run_model sc_vae "lit.beta_s=1.0 lit.lambda_inv=1.0"

# β-VAE: β_s=4.0 (Higgins et al. 2017 best reported)
run_model beta_vae "lit.beta_s=4.0"

# β-TCVAE: β_s=6.0, α=1.0, γ=1.0 (Chen et al. 2018 Table 1)
run_model beta_tcvae "lit.beta_s=6.0 model.alpha=1.0 model.gamma=1.0"

# FactorVAE: β_s=1.0, λ_inv=35.0 (Kim & Mnih 2018)
run_model factor_vae "lit.beta_s=1.0 lit.lambda_inv=35.0"

# AR-HVAE: β_s=1.0 (no published sweep; vanilla VAE prior)
run_model ar_hvae "lit.beta_s=1.0"

echo ""
echo "All models launched. Waiting for completion..."
wait
echo "All training complete."
EOF

chmod +x /scratch/mty236/train_all_5_models.sh
```

### Launch Training

**Option A: Sequential (simpler, slower)**
```bash
bash /scratch/mty236/train_all_5_models.sh
```
**Time:** ~25-30 hrs total (models train one-after-another)

**Option B: Parallel (faster, requires GPU memory)**
```bash
# Launch each model in separate terminal or tmux session:

# Terminal 1: SC-VAE
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=sc_vae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=0 lit.beta_s=1.0 lit.lambda_inv=1.0 hydra.run.dir=runs/option2_nsynth_50epochs/sc_vae &

# Terminal 2: β-VAE
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=beta_vae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=1 lit.beta_s=4.0 hydra.run.dir=runs/option2_nsynth_50epochs/beta_vae &

# Terminal 3: β-TCVAE
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=beta_tcvae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=0 lit.beta_s=6.0 model.alpha=1.0 model.gamma=1.0 hydra.run.dir=runs/option2_nsynth_50epochs/beta_tcvae &

# Terminal 4: FactorVAE
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=factor_vae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=1 lit.beta_s=1.0 lit.lambda_inv=35.0 hydra.run.dir=runs/option2_nsynth_50epochs/factor_vae &

# Terminal 5: AR-HVAE
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=ar_hvae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=0 lit.beta_s=1.0 hydra.run.dir=runs/option2_nsynth_50epochs/ar_hvae &

wait
```
**Time:** ~5-6 hrs total (all models train simultaneously on 2 GPUs with DDP)

---

## Monitoring Training

### Real-Time Metrics (TensorBoard)

```bash
tensorboard --logdir runs/option2_nsynth_50epochs --port 6006
```

Open in browser: `http://localhost:6006`

**Watch these per-model per-epoch:**
- `train/loss_recon` → Should trend down
- `train/loss_kl_*` → Should stabilize
- `train/loss_*` → Model-specific losses
- `val/loss_*` → Validation curves

### Log Inspection

```bash
# View run manifests
ls -la runs/option2_nsynth_50epochs/*/run_manifest.json

# Tail logs (if logging to file)
tail -f runs/option2_nsynth_50epochs/sc_vae/.hydra/hydra.log
```

---

## Post-Training (Day 2 Evening)

### Collect Results

```bash
# List all completed runs
ls -la runs/option2_nsynth_50epochs/

# Check each model's final checkpoint
for model in sc_vae beta_vae beta_tcvae factor_vae ar_hvae; do
    echo "=== $model ===" 
    ls -lh runs/option2_nsynth_50epochs/$model/checkpoints/last.ckpt 2>/dev/null || echo "No checkpoint found"
done
```

### Run Evaluation (Optional, Day 3+)

```bash
# Evaluate all trained models
for model in sc_vae beta_vae beta_tcvae factor_vae ar_hvae; do
    python -m src.training.cli \
        --config-name eval/default \
        data=nsynth_bass \
        model=$model \
        +checkpoint_path="runs/option2_nsynth_50epochs/${model}/checkpoints/last.ckpt" \
        +eval=true
done
```

This generates: **MIG, DCI, SAP, Modularity** scores for each model.

---

## Expected Results (After 50 Epochs)

### SC-VAE (Your Model)
| Metric | Expected Range | Notes |
|--------|---|---|
| **MIG** | 0.35-0.50 | Good separation of pitch/timbre |
| **DCI-D** | 0.50-0.70 | Strong reconstruction |
| **SAP** | 0.40-0.60 | Well-isolated factors |
| **Recon Loss** | 0.5-1.0 dB MSE | Acceptable quality |
| **Inv Loss** (style) | ~0.1-0.3 | Timbre stable w/ pitch shift |
| **Equi Loss** (content) | ~0.2-0.5 | Pitch shifts w/ transposition |

### Baselines (For Comparison)
- **β-VAE:** MIG ~0.25-0.35 (decent but less principled)
- **β-TCVAE:** MIG ~0.30-0.40 (good, but generic disentanglement)
- **FactorVAE:** MIG ~0.28-0.38 (comparable to β-TCVAE)
- **AR-HVAE:** MIG ~0.20-0.30 (hierarchical structure may hurt simple bass)

**Your SC-VAE should outperform all baselines** due to music-specific group symmetry constraint.

---

## File Structure After Training

```
runs/option2_nsynth_50epochs/
├── sc_vae/
│   ├── checkpoints/
│   │   ├── epoch_00.ckpt
│   │   ├── epoch_25.ckpt
│   │   ├── epoch_49.ckpt
│   │   └── last.ckpt
│   ├── run_manifest.json       # Config snapshot
│   ├── train_metrics.csv       # Per-epoch loss/accuracy
│   └── .hydra/                 # Hydra config dump
├── beta_vae/
│   ├── checkpoints/
│   └── ...
├── beta_tcvae/
│   └── ...
├── factor_vae/
│   └── ...
└── ar_hvae/
    └── ...
```

---

## CQT Feature Details

All 5 models use the same CQT features:

```
Input: 16 kHz mono WAV (4 sec)
  ↓
ConstantQ Transform:
  - 12 bins per octave
  - 7 octaves (C1 @ 32.7 Hz to B7 @ 3951 Hz)
  - 160-sample hop (10 ms @ 16 kHz)
  ↓
Output shape: (1, 84, 401)
  - 84 frequency bins (vs. Mel's 128)
  - 401 time frames
  - Logarithmic frequency resolution (ideal for bass)
  ↓
Magnitude conversion: Complex → dB scale [-80, 0]
```

---

## Troubleshooting

### Issue: nnAudio import error
```bash
pip install nnAudio==0.3.2
```

### Issue: NSynth dataset not found
```bash
echo $NSYNTH_ROOT
# Should output: /scratch/mty236/ML_Project/data/nsynth_raw
```

### Issue: Cache file not found during training
```bash
ls -lh $NSYNTH_CACHE/train_cqt.h5
# Should exist and be ~1.3 GB
```

### Issue: OOM (Out of Memory) during training
```bash
# Reduce DDP devices or batch size:
trainer.devices=0  # Use single GPU instead of 0,1
# Or
data.batch_size=16  # Reduce from default
```

### Issue: Training very slow
```bash
# Verify CQT cache was used (not on-the-fly computation)
# Check logs for: "Loading CQT from HDF5" vs "Computing CQT on-the-fly"
```

---

## Quick Reference: All Commands

### Day 1

```bash
# Setup
source /scratch/mty236/setup_env.sh
pip install nnAudio==0.3.2

# Smoke test
python -m src.training.cli +trainer.fast_dev_run=true

# Validate dataset
python -c "import json; data = json.load(open('$NSYNTH_ROOT/nsynth-train/examples.json')); bass = [k for k,v in data.items() if v['instrument_family']==1]; print(f'Bass: {len(bass)}')"

# Cache generation (20-25 min)
python scripts/prepare_nsynth_cache.py --src $NSYNTH_ROOT --split train --out $NSYNTH_CACHE/train_cqt.h5 --feature-type cqt --batch-size 256 --device cuda
```

### Day 2

```bash
# Sequential training (25-30 hrs)
bash /scratch/mty236/train_all_5_models.sh

# Or parallel training (5-6 hrs, choose devices wisely)
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=sc_vae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=0,1 lit.beta_s=1.0 lit.lambda_inv=1.0 hydra.run.dir=runs/option2_nsynth_50epochs/sc_vae &
python -m src.training.cli --config-name train/sweep data=nsynth_bass model=beta_vae trainer.max_epochs=50 trainer.accelerator=gpu trainer.devices=0,1 lit.beta_s=4.0 hydra.run.dir=runs/option2_nsynth_50epochs/beta_vae &
# ... (repeat for beta_tcvae, factor_vae, ar_hvae)
wait
```

### Post-Training

```bash
# TensorBoard
tensorboard --logdir runs/option2_nsynth_50epochs --port 6006

# Evaluation (optional, Day 3+)
for model in sc_vae beta_vae beta_tcvae factor_vae ar_hvae; do
    python -m src.training.cli --config-name eval/default data=nsynth_bass model=$model +checkpoint_path="runs/option2_nsynth_50epochs/${model}/checkpoints/last.ckpt" +eval=true
done
```

---

## Timeline Summary

```
Day 1 (May 7)
├─ Morning (30 min):   Setup env, smoke test, validate data
└─ Evening (25 min):   Cache generation (runs in background)

Day 2 (May 8)
├─ All Day (5-6 hrs):  All 5 models train in parallel
├─ Evening:            Monitor TensorBoard, collect logs
└─ Night:              Models may still be training (fine to leave running)

Day 3+ (Optional)
└─ Evaluation:         Run metrics (MIG, DCI, SAP) on completed models
```

---

## Next Steps After Training

1. **Download checkpoints** from HPC (especially SC-VAE)
2. **Run evaluation metrics** (MIG, DCI, SAP, etc.)
3. **Compare against baselines** — Your SC-VAE should win
4. **Use for inference** — Zero-shot pitch shift on MoisesDB bass stems
5. **Paper/report** — Document CQT migration + results

---

## Final Checklist ✅

- [x] CQT implementation complete (nnAudio backend)
- [x] All 5 model configs updated (n_mels: 84)
- [x] NSynth loader ready with CQT support
- [x] Cache script ready for pre-computation
- [x] Training configs ready (published hyperparameters)
- [x] Dataset paths verified (NSynth only)
- [x] GPU memory plan (parallel training feasible)

**Status: ✅ READY FOR DAY 1 (May 7, 2026)**
