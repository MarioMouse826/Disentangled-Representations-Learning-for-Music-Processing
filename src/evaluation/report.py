"""Aggregate assessment report — LaTeX table + DCI heatmap + Pareto scatter.

Inputs are lists of "run" dicts:
    {
        "model": "sc_vae",
        "seed": 0,
        "metrics": {"mig": 0.6, "sap": 0.5, "srr": 12.0, ...},
        "dci_importance": (D, K) np.ndarray,   # optional
    }

Outputs:
    - LaTeX table (rows=models, cols=metrics) with mean ± std.
    - DCI importance heatmaps (one PDF per model).
    - Pareto scatter (x=SRR, y=SAP by default) across all runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Aggregation


def aggregate_results(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Group runs by model; per metric compute mean / std / n.

    Returns: `{model_name: {metric_name: {"mean": ..., "std": ..., "n": ...}}}`.
    """
    if not runs:
        raise ValueError("runs must be non-empty")
    by_model: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        m = r["model"]
        metrics = r.get("metrics", {})
        slot = by_model.setdefault(m, {})
        for name, val in metrics.items():
            slot.setdefault(name, []).append(float(val))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model, metric_map in by_model.items():
        row: dict[str, dict[str, float]] = {}
        for name, vals in metric_map.items():
            arr = np.asarray(vals, dtype=np.float64)
            row[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }
        summary[model] = row
    return summary


# ---------------------------------------------------------------------------
# LaTeX formatting


def format_cell(mean: float, std: float, *, precision: int = 3, n: int = 2) -> str:
    """Render a mean±std cell; drop ± when only one seed."""
    if n <= 1:
        return f"${mean:.{precision}f}$"
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def _escape_model_name(name: str) -> str:
    return name.replace("_", "\\_")


def _metric_header(name: str) -> str:
    # Map known metrics to pretty column names; pass through unknown.
    pretty = {
        "mig": "MIG",
        "sap": "SAP",
        "modularity": "Mod",
        "factor_vae_score": "FVAE",
        "dci_d": "D",
        "dci_c": "C",
        "dci_i": "I",
        "srr": "SRR (dB)",
        "invariance_ratio": "IR",
        "equivariance_ratio": "ER",
        "pitch_acc_50": "PitchAcc@50",
    }
    return pretty.get(name, name.replace("_", " ").upper())


def write_latex_table(
    *,
    runs: Sequence[dict[str, Any]],
    out_path: Path,
    metric_order: Sequence[str],
    model_order: Sequence[str] | None = None,
    precision: int = 3,
    caption: str = "Disentanglement + reconstruction metrics (mean $\\pm$ std across seeds).",
    label: str = "tab:results",
) -> None:
    """Render a booktabs-style LaTeX table to `out_path`."""
    agg = aggregate_results(runs)
    models = list(model_order) if model_order else sorted(agg.keys())

    col_spec = "l" + "c" * len(metric_order)
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["Model"] + [_metric_header(m) for m in metric_order]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for model in models:
        row_cells = [_escape_model_name(model)]
        row = agg.get(model, {})
        for metric in metric_order:
            entry = row.get(metric)
            if entry is None:
                row_cells.append("--")
            else:
                row_cells.append(
                    format_cell(
                        entry["mean"], entry["std"],
                        precision=precision, n=entry["n"],
                    )
                )
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Pareto


def pareto_front(points: np.ndarray, *, maximize: bool = True) -> np.ndarray:
    """Boolean mask: True at non-dominated points.

    Args:
        points: `(N, 2)` — columns are the two axes to compare.
        maximize: True for "higher is better" on both axes; False for
            "lower is better". For mixed, flip one column sign upstream.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    if maximize:
        for i in range(n):
            # Dominated if some j has both coords >= and at least one strictly >.
            greater_equal = np.all(points >= points[i], axis=1)
            strictly_greater = np.any(points > points[i], axis=1)
            if np.any(greater_equal & strictly_greater):
                mask[i] = False
    else:
        for i in range(n):
            less_equal = np.all(points <= points[i], axis=1)
            strictly_less = np.any(points < points[i], axis=1)
            if np.any(less_equal & strictly_less):
                mask[i] = False
    return mask


# ---------------------------------------------------------------------------
# Figures (matplotlib-backed)


def _lazy_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_dci_heatmap(
    *,
    importance_matrix: np.ndarray,
    factor_names: Sequence[str],
    out_path: Path,
    title: str | None = None,
) -> None:
    """Heatmap of DCI feature-importance (rows=latent dims, cols=factors)."""
    plt = _lazy_mpl()
    r = np.asarray(importance_matrix)
    if r.ndim != 2 or r.shape[1] != len(factor_names):
        raise ValueError(
            f"importance matrix shape {r.shape} vs factor_names len "
            f"{len(factor_names)} mismatch"
        )
    fig, ax = plt.subplots(figsize=(0.6 * max(3, len(factor_names)) + 1, 0.25 * r.shape[0] + 1))
    try:
        im = ax.imshow(r, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(r.shape[0]))
        ax.set_yticklabels([f"z_{i}" for i in range(r.shape[0])], fontsize=7)
        ax.set_xticks(np.arange(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=30, ha="right")
        ax.set_xlabel("factor")
        ax.set_ylabel("latent dim")
        if title is not None:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, label="importance")
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)


def write_pareto_scatter(
    *,
    runs: Iterable[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    out_path: Path,
    maximize: tuple[bool, bool] = (True, True),
) -> None:
    """Scatter of runs with the Pareto front highlighted.

    Each run is drawn at `(metrics[x_metric], metrics[y_metric])`, colored
    by model. Runs missing either metric are skipped.
    """
    plt = _lazy_mpl()
    by_model: dict[str, list[tuple[float, float]]] = {}
    all_points: list[tuple[float, float, str]] = []
    for r in runs:
        metrics = r.get("metrics", {})
        if x_metric not in metrics or y_metric not in metrics:
            continue
        x, y = float(metrics[x_metric]), float(metrics[y_metric])
        by_model.setdefault(r["model"], []).append((x, y))
        all_points.append((x, y, r["model"]))

    if not all_points:
        raise ValueError("no runs contain both metrics")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    try:
        for model, pts in by_model.items():
            arr = np.asarray(pts, dtype=np.float64)
            ax.scatter(arr[:, 0], arr[:, 1], label=model, s=28, alpha=0.8)

        pts_arr = np.asarray([(p[0], p[1]) for p in all_points], dtype=np.float64)
        # Interpret `maximize` per axis by flipping sign before pareto_front.
        signed = pts_arr.copy()
        if not maximize[0]:
            signed[:, 0] = -signed[:, 0]
        if not maximize[1]:
            signed[:, 1] = -signed[:, 1]
        mask = pareto_front(signed, maximize=True)

        front = pts_arr[mask]
        order = np.argsort(front[:, 0])
        front = front[order]
        ax.plot(front[:, 0], front[:, 1], linestyle="--", linewidth=1.2, color="k",
                label="Pareto front")

        ax.set_xlabel(x_metric.upper())
        ax.set_ylabel(y_metric.upper())
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical significance (plan Task 6.4)


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> dict[str, float]:
    """Paired bootstrap on `a - b`. Resamples paired indices, computes mean
    of the differences, and returns percentile CI at level `1 - alpha`.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(
            f"paired samples must share length, got {a.size} vs {b.size}"
        )
    if a.size == 0:
        raise ValueError("cannot bootstrap empty samples")
    diff = a - b
    rng = np.random.default_rng(random_state)
    n = diff.size
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi}


def paired_ttest_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test p-value on `a - b`."""
    from scipy.stats import ttest_rel

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(
            f"paired samples must share length, got {a.size} vs {b.size}"
        )
    # Handle zero-variance differences: ttest_rel returns NaN; clamp to 1.0
    # so downstream code treats it as "not significant" rather than crashing.
    diff = a - b
    if diff.size < 2 or float(diff.std(ddof=1)) == 0.0:
        return 1.0
    res = ttest_rel(a, b)
    p = float(res.pvalue)
    return p if np.isfinite(p) else 1.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta — non-parametric effect size in [-1, 1].

        delta = (#(a_i > b_j) - #(a_i < b_j)) / (n_a * n_b)

    Values: |delta| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("cannot compute Cliff's delta on empty samples")
    # Vectorized pairwise sign counting — O(n_a * n_b) memory, fine for
    # seed-count arrays (n <= 20 typically).
    diff = a[:, None] - b[None, :]
    gt = int((diff > 0).sum())
    lt = int((diff < 0).sum())
    return float((gt - lt) / (a.size * b.size))


def _seed_series(
    runs: Sequence[dict[str, Any]], model: str, metric: str
) -> np.ndarray:
    """Values for `metric` across seeds of `model`, sorted by seed."""
    pairs: list[tuple[int, float]] = []
    for r in runs:
        if r.get("model") != model:
            continue
        m = r.get("metrics", {})
        if metric not in m:
            continue
        pairs.append((int(r.get("seed", -1)), float(m[metric])))
    pairs.sort(key=lambda p: p[0])
    return np.asarray([v for _, v in pairs], dtype=np.float64)


def compare_models(
    *,
    runs: Sequence[dict[str, Any]],
    metric: str,
    baseline: str,
    candidate: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> dict[str, float]:
    """Paired comparison of `candidate` vs `baseline` on `metric`.

    Pairing is by seed — the seed lists for both models must match in length
    and are assumed to cover the same seed values.

    Returns: {delta_mean, ci_low, ci_high, p_value, cliffs_delta}.
    """
    cand = _seed_series(runs, candidate, metric)
    base = _seed_series(runs, baseline, metric)
    if cand.size == 0 or base.size == 0:
        raise ValueError(
            f"no runs for metric {metric!r} on {candidate!r}/{baseline!r}"
        )
    if cand.size != base.size:
        raise ValueError(
            f"seed count mismatch: {candidate}={cand.size} vs "
            f"{baseline}={base.size}; pass matching seed sets"
        )
    bs = paired_bootstrap_diff(
        cand, base, n_boot=n_boot, alpha=alpha, random_state=random_state
    )
    p = paired_ttest_pvalue(cand, base)
    d = cliffs_delta(cand, base)
    return {
        "delta_mean": bs["mean_diff"],
        "ci_low": bs["ci_low"],
        "ci_high": bs["ci_high"],
        "p_value": p,
        "cliffs_delta": d,
    }
