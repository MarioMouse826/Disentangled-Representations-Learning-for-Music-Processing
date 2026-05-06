#!/usr/bin/env bash
# Ablation matrix runner (plan Task 6.3).
#
# First pass on NSynth-bass with a single seed; top-3 are re-run on
# MoisesDB with 5 seeds (driven by SECOND_PASS=1 env var).
#
# Invocation:
#   bash scripts/run_ablations.sh                  # first pass (NSynth, seed 0)
#   SECOND_PASS=1 bash scripts/run_ablations.sh    # re-run top-3 on MoisesDB
#
# Env overrides:
#   EXP, OUT_DIR, DATASET, SEED, MODEL
set -euo pipefail

EXP="${EXP:-ablations_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-runs/ablations/${EXP}}"
DATASET="${DATASET:-nsynth_bass}"
MODEL="${MODEL:-sc_vae}"
SEED="${SEED:-0}"
SECOND_PASS="${SECOND_PASS:-0}"

mkdir -p "${OUT_DIR}"
echo "[run_ablations] out=${OUT_DIR}  dataset=${DATASET}  model=${MODEL}"

# Each entry: "<ablation_id>|<hydra overrides>"
# IMPLEMENTED — can run today against the existing codebase.
ABLATIONS_READY=(
  "full|"
  "no_l_inv|lit.lambda_inv=0"
  "no_l_equi|lit.lambda_equi=0"
  "no_l_swap|lit.lambda_swap=0"
  "rep_identity|model.rep_type=identity"
  "rep_translation|model.rep_type=translation"
  "dims_swapped|model.d_s=16 model.d_c=64"
  "dims_unswapped|model.d_s=64 model.d_c=16"
  "beta_const|lit.beta_schedule=constant lit.beta_s=4 lit.beta_c=4"
  "beta_cyclical|lit.beta_schedule=cyclical lit.beta_s=4 lit.beta_c=4"
  "beta_asym_high_c|lit.beta_s=1 lit.beta_c=8"
  "beta_asym_high_s|lit.beta_s=8 lit.beta_c=1"
  "unpaired_batch|data.paired=false"
  "aug_uniform_half|lit.g_cents_dist=uniform_half"
  "aug_discrete|lit.g_cents_dist=discrete_semitones"
)

# PLANNED — require code not yet merged; tracked as TODO for Phase 7+.
# Uncomment individual entries once the corresponding feature lands.
ABLATIONS_TODO=(
  # "aug_phase_vocoder|data.aug_backend=phase_vocoder"     # Task 6.3 row 6
  # "enc_raw_waveform|model.encoder=raw_wavenet"           # Task 6.3 row 7
  # "dec_ddsp|model.decoder=ddsp"                          # Task 7.1
  # "pitch_embedding|model.pitch_embed=learned"            # Task 6.3 row 13
  # "synth_pretrain|lit.pretrain_synth=true"               # Task 6.3 row 12
)

# Second pass — top-3 on MoisesDB × 5 seeds. Override TOP_3 at invocation.
TOP_3="${TOP_3:-full no_l_equi rep_identity}"
SECOND_SEEDS=(0 1 2 3 4)
SECOND_DATASET="${SECOND_DATASET:-moisesdb_bass}"

run_one() {
  local id="$1"
  local overrides="$2"
  local seed="$3"
  local dataset="$4"
  local run_dir="${OUT_DIR}/${id}_seed${seed}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[skip] ${id} seed=${seed} (metrics.json exists)"
    return
  fi
  echo "[run] ${id} seed=${seed} dataset=${dataset} overrides='${overrides}'"
  mkdir -p "${run_dir}"
  # shellcheck disable=SC2086
  python -m src.training.cli \
    --config-name sweep \
    model="${MODEL}" \
    data="${dataset}" \
    seed="${seed}" \
    hydra.run.dir="${run_dir}" \
    ${overrides}
}

if [[ "${SECOND_PASS}" == "0" ]]; then
  for entry in "${ABLATIONS_READY[@]}"; do
    id="${entry%%|*}"
    overrides="${entry#*|}"
    run_one "${id}" "${overrides}" "${SEED}" "${DATASET}"
  done
  echo "[run_ablations] first pass complete."
  echo "[next] pick top-3 from runs, re-run as:"
  echo "  TOP_3='full no_l_equi rep_identity' SECOND_PASS=1 bash scripts/run_ablations.sh"
else
  for id in ${TOP_3}; do
    # Look up overrides from the ready list.
    found=""
    for entry in "${ABLATIONS_READY[@]}"; do
      if [[ "${entry%%|*}" == "${id}" ]]; then
        found="${entry#*|}"
        break
      fi
    done
    if [[ -z "${found}" && "${id}" != "full" ]]; then
      echo "[warn] top-3 id '${id}' not found in ABLATIONS_READY; skipping"
      continue
    fi
    for seed in "${SECOND_SEEDS[@]}"; do
      run_one "${id}" "${found}" "${seed}" "${SECOND_DATASET}"
    done
  done
  echo "[run_ablations] second pass complete (5 seeds on ${SECOND_DATASET})."
fi
