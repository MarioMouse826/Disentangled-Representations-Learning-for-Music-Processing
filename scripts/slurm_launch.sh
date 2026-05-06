#!/usr/bin/env bash
# SLURM launch script for SC-VAE training on NYU HPC (NVIDIA L40, 46 GB).
#
# Usage:
#   sbatch scripts/slurm_launch.sh stage1
#   sbatch scripts/slurm_launch.sh stage234
#   sbatch scripts/slurm_launch.sh stage56
#
# Required env vars (set in your ~/.bashrc or pass via --export):
#   NSYNTH_CACHE_TRAIN  — path to NSynth train.h5
#   NSYNTH_CACHE_VALID  — path to NSynth valid.h5
#   MOISESDB_ROOT       — path to MoisesDB dataset root
#   CREPE_CACHE         — path to CREPE f0 cache HDF5
#   PROJECT_ROOT        — absolute path to this repo

#SBATCH --job-name=sc_vae
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

set -euo pipefail

STAGE="${1:-stage1}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"

cd "$PROJECT_ROOT"
mkdir -p logs

# Load modules and activate conda env (NYU HPC torch cluster).
module purge 2>/dev/null || true
module load anaconda3/2025.06
module load cuda/12.1 2>/dev/null || true

export CONDA_ENVS_PATH=/scratch/ig2671/conda_envs
export CONDA_PKGS_DIRS=/scratch/ig2671/conda_pkgs
eval "$(conda shell.bash hook)"
conda activate sc_vae

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[slurm] Stage=$STAGE  Node=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
case "$STAGE" in

  stage1)
    # Pilot run: sc_vae, seed=0, 20 epochs on NSynth-bass.
    python -m src.training.cli \
      model=sc_vae \
      data=nsynth_bass \
      seed=0 \
      trainer.max_epochs=20 \
      trainer.precision=bf16-mixed \
      lit.beta_schedule=cyclical \
      lit.n_cycles=4 \
      lit.beta_peak=4.0
    ;;

  stage234)
    # Main results (Stage 2) + ablations (Stage 3) + MoisesDB mini-arm (Stage 4).

    # Stage 2: sc_vae + beta_vae + beta_tcvae, seeds {0,1,2}, 100 epochs.
    python -m src.training.cli --config-name=sweep --multirun \
      data=nsynth_bass \
      trainer.max_epochs=100 \
      trainer.precision=bf16-mixed \
      model=sc_vae \
      seed=0,1,2 \
      lit.beta_schedule=cyclical \
      lit.n_cycles=4 \
      lit.beta_peak=4.0

    python -m src.training.cli --config-name=sweep --multirun \
      data=nsynth_bass \
      trainer.max_epochs=100 \
      trainer.precision=bf16-mixed \
      model=beta_vae \
      seed=0,1,2 \
      lit.beta_s=4.0 \
      lit.beta_schedule=cyclical \
      lit.n_cycles=4 \
      lit.beta_peak=4.0

    python -m src.training.cli --config-name=sweep --multirun \
      data=nsynth_bass \
      trainer.max_epochs=100 \
      trainer.precision=bf16-mixed \
      model=beta_tcvae \
      seed=0,1,2 \
      lit.beta_s=6.0 \
      lit.beta_schedule=cyclical \
      lit.n_cycles=4 \
      lit.beta_peak=4.0

    # Stage 3: ablations (sc_vae only).
    bash scripts/run_ablations.sh

    # Stage 4: MoisesDB mini-arm.
    python -m src.training.cli \
      model=sc_vae \
      data=moisesdb_bass \
      seed=0 \
      trainer.max_epochs=50 \
      trainer.precision=bf16-mixed \
      lit.beta_schedule=cyclical
    ;;

  stage56)
    # Stage 5: zero-shot OOD eval (inference only).
    # Stage 6: full metric rollup.
    bash scripts/run_full_eval.sh
    ;;

  *)
    echo "[slurm] Unknown stage: $STAGE. Use stage1 | stage234 | stage56." >&2
    exit 1
    ;;
esac

echo "[slurm] Done. Stage=$STAGE"
