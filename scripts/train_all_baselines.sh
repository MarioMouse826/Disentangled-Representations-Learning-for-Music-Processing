#!/usr/bin/env bash
# Train all baseline models with published hyperparameters (training plan P4).
# Runs on NSynth-bass, 3 seeds, 100 epochs each.
#
# Published β values (Higgins 2017, Chen 2018, Kim & Mnih 2018):
#   beta_vae    β=4.0
#   beta_tcvae  β=6.0, α=1.0, γ=1.0
#   factor_vae  γ=35.0
#   ar_hvae     β=1.0 (no published sweep; using default)
#
# Usage:
#   bash scripts/train_all_baselines.sh                         # default: 3 seeds, 100 epochs
#   SEEDS="0" MAX_EPOCHS=20 bash scripts/train_all_baselines.sh # smoke run
#   DATA=synthetic bash scripts/train_all_baselines.sh          # synthetic data
#
# Env overrides:
#   SEEDS        space-separated list (default: "0 1 2")
#   MAX_EPOCHS   (default: 100)
#   DATA         Hydra data config name (default: nsynth_bass)
#   OUT_ROOT     run output root (default: runs/baselines)

set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
DATA="${DATA:-nsynth_bass}"
OUT_ROOT="${OUT_ROOT:-runs/baselines}"

echo "[train_all_baselines] data=${DATA}  max_epochs=${MAX_EPOCHS}  seeds=(${SEEDS})"
echo "[train_all_baselines] outputs → ${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"

# ---------------------------------------------------------------------------
run_baseline() {
  local model="$1"
  local seed="$2"
  local extra_overrides="$3"
  local run_dir="${OUT_ROOT}/${model}_seed${seed}"

  if [[ -f "${run_dir}/run_manifest.json" ]]; then
    echo "[skip] ${model} seed=${seed} (run_manifest.json exists)"
    return
  fi

  echo "[train] ${model}  seed=${seed}  overrides='${extra_overrides}'"
  mkdir -p "${run_dir}"

  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  python -m src.training.cli \
    --config-name train/sweep \
    model="${model}" \
    data="${DATA}" \
    seed="${seed}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    trainer.precision=bf16-mixed \
    hydra.run.dir="${run_dir}" \
    ${extra_overrides}
}

# ---------------------------------------------------------------------------
for seed in ${SEEDS}; do

  # β-VAE: β=4.0 (Higgins 2017 best reported on dSprites)
  run_baseline beta_vae "${seed}" "lit.beta_s=4.0"

  # β-TCVAE: β=6.0, α=1.0, γ=1.0 (Chen 2018 Table 1)
  # model.alpha/gamma already defined in configs/model/beta_tcvae.yaml; no + needed.
  run_baseline beta_tcvae "${seed}" "lit.beta_s=6.0 model.alpha=1.0 model.gamma=1.0"

  # FactorVAE: β=1.0, γ=35.0 (Kim & Mnih 2018). γ routes through lit.lambda_inv.
  run_baseline factor_vae "${seed}" "lit.beta_s=1.0 lit.lambda_inv=35.0"

  # AR-HVAE: no published sweep — use β=1.0 (vanilla VAE prior)
  run_baseline ar_hvae "${seed}" "lit.beta_s=1.0"

done

echo ""
echo "[train_all_baselines] All baselines complete."
echo "Run evaluation:"
echo "  bash scripts/run_full_eval.sh"
