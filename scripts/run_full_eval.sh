#!/usr/bin/env bash
# Full evaluation sweep: run each model across 5 seeds, collect metrics,
# emit LaTeX table + DCI heatmaps + Pareto-front scatter.
#
# Expects Hydra configs under configs/ and a trained checkpoint per model.
# Writes outputs to runs/<exp>/eval/.
set -euo pipefail

EXP="${EXP:-full_eval_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-runs/${EXP}/eval}"
SEEDS=(0 1 2 3 4)
MODELS=(beta_vae beta_tcvae factor_vae ar_hvae sc_vae)
DATASET="${DATASET:-nsynth_bass}"

mkdir -p "${OUT_DIR}"

echo "[run_full_eval] output: ${OUT_DIR}"
echo "[run_full_eval] models: ${MODELS[*]}"
echo "[run_full_eval] seeds: ${SEEDS[*]}"

# 1. Per-model x per-seed evaluation -> writes one JSON per run.
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_json="${OUT_DIR}/${model}_seed${seed}.json"
    if [[ -f "${run_json}" ]]; then
      echo "[skip] ${run_json} exists"
      continue
    fi
    echo "[eval] ${model} seed=${seed}"
    python -m src.training.cli \
      --config-name base \
      +mode=eval \
      model="${model}" \
      data="${DATASET}" \
      seed="${seed}" \
      eval.out_json="${run_json}"
  done
done

# 2. Aggregate to LaTeX + figures.
python - <<PY
from pathlib import Path
import json
import numpy as np
from src.evaluation.report import (
    write_latex_table, write_dci_heatmap, write_pareto_scatter,
)

out_dir = Path("${OUT_DIR}")
runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_seed*.json"))]
if not runs:
    raise SystemExit("[run_full_eval] no run JSONs found - aborting aggregate step")

metric_order = [
    "mig", "sap", "modularity", "factor_vae_score",
    "dci_d", "dci_c", "dci_i",
    "srr", "invariance_ratio", "equivariance_ratio", "pitch_acc_50",
]
model_order = ["beta_vae", "beta_tcvae", "factor_vae", "ar_hvae", "sc_vae"]

write_latex_table(
    runs=runs,
    out_path=out_dir / "results.tex",
    metric_order=metric_order,
    model_order=model_order,
)

# One heatmap per model using the first-seed run's DCI importance matrix.
fig_dir = Path("docs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
seen: set[str] = set()
for r in runs:
    model = r["model"]
    if model in seen or "dci_importance" not in r:
        continue
    R = np.asarray(r["dci_importance"])
    write_dci_heatmap(
        importance_matrix=R,
        factor_names=r.get("factor_names", [f"v{i}" for i in range(R.shape[1])]),
        out_path=fig_dir / f"dci_heatmap_{model}.pdf",
        title=model,
    )
    seen.add(model)

write_pareto_scatter(
    runs=runs, x_metric="srr", y_metric="sap",
    out_path=fig_dir / "pareto_srr_sap.pdf",
)
print(f"[ok] wrote {out_dir}/results.tex and {fig_dir}/*.pdf")
PY
