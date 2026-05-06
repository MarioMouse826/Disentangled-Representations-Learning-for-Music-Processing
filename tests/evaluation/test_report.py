"""Tests for the aggregate assessment report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.report import (
    aggregate_results,
    format_cell,
    pareto_front,
    write_dci_heatmap,
    write_latex_table,
    write_pareto_scatter,
)


# -- aggregate_results ----------------------------------------------------


def _run(model: str, seed: int, **metrics: float) -> dict:
    return {"model": model, "seed": seed, "metrics": metrics}


def test_aggregate_groups_by_model_and_computes_mean_std() -> None:
    runs = [
        _run("beta_vae", 0, mig=0.30, sap=0.20),
        _run("beta_vae", 1, mig=0.32, sap=0.22),
        _run("sc_vae", 0, mig=0.60, sap=0.50),
        _run("sc_vae", 1, mig=0.62, sap=0.48),
    ]
    out = aggregate_results(runs)
    assert set(out.keys()) == {"beta_vae", "sc_vae"}
    b = out["beta_vae"]
    assert "mig" in b and "sap" in b
    assert abs(b["mig"]["mean"] - 0.31) < 1e-9
    assert b["mig"]["n"] == 2
    # std is sample std (ddof=1).
    assert abs(b["mig"]["std"] - float(np.std([0.30, 0.32], ddof=1))) < 1e-9


def test_aggregate_handles_missing_metrics_per_run() -> None:
    runs = [
        _run("a", 0, mig=0.1),
        _run("a", 1, mig=0.2, srr=10.0),
    ]
    out = aggregate_results(runs)
    assert out["a"]["mig"]["n"] == 2
    assert out["a"]["srr"]["n"] == 1


def test_aggregate_rejects_empty_runs() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_results([])


# -- format_cell ----------------------------------------------------------


def test_format_cell_with_std() -> None:
    assert format_cell(0.3145, 0.0123, precision=3) == "$0.315 \\pm 0.012$"


def test_format_cell_single_seed_drops_std() -> None:
    assert format_cell(0.5, 0.0, precision=2, n=1) == "$0.50$"


# -- pareto_front ---------------------------------------------------------


def test_pareto_front_identifies_non_dominated() -> None:
    # Maximize both axes. Point (3, 3) dominates (2, 2); (3, 1) not dominated
    # by (2, 2) on y but dominated on x only.
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [3.0, 1.0], [0.0, 4.0]])
    mask = pareto_front(pts, maximize=True)
    # Non-dominated: (3, 3) and (0, 4).
    assert mask.tolist() == [False, False, True, False, True]


def test_pareto_front_minimize_direction() -> None:
    pts = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
    # Minimize both → front is (1,3), (2,2), (3,1).
    mask = pareto_front(pts, maximize=False)
    assert mask.tolist() == [True, True, True, False]


def test_pareto_front_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        pareto_front(np.array([1.0, 2.0, 3.0]), maximize=True)


# -- LaTeX + figures -------------------------------------------------------


def test_write_latex_table_writes_file(tmp_path: Path) -> None:
    runs = [
        _run("beta_vae", 0, mig=0.3, sap=0.2, srr=10.0),
        _run("beta_vae", 1, mig=0.32, sap=0.22, srr=10.5),
        _run("sc_vae", 0, mig=0.6, sap=0.5, srr=12.0),
        _run("sc_vae", 1, mig=0.62, sap=0.48, srr=12.5),
    ]
    path = tmp_path / "results.tex"
    write_latex_table(
        runs=runs,
        out_path=path,
        metric_order=["mig", "sap", "srr"],
        model_order=["beta_vae", "sc_vae"],
    )
    content = path.read_text()
    assert "\\begin{tabular}" in content
    assert "\\end{tabular}" in content
    assert "beta\\_vae" in content or "beta_vae" in content
    assert "MIG" in content.upper()


def test_write_dci_heatmap_writes_pdf(tmp_path: Path) -> None:
    R = np.random.default_rng(0).random((6, 3))
    out_path = tmp_path / "heatmap.pdf"
    write_dci_heatmap(
        importance_matrix=R,
        factor_names=["pitch", "instrument", "velocity"],
        out_path=out_path,
        title="sc_vae",
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_write_pareto_scatter_writes_pdf(tmp_path: Path) -> None:
    runs = [
        _run("beta_vae", 0, srr=10.0, sap=0.2),
        _run("beta_vae", 1, srr=11.0, sap=0.22),
        _run("sc_vae", 0, srr=12.0, sap=0.5),
        _run("sc_vae", 1, srr=12.5, sap=0.55),
    ]
    out_path = tmp_path / "pareto.pdf"
    write_pareto_scatter(
        runs=runs, x_metric="srr", y_metric="sap", out_path=out_path
    )
    assert out_path.exists() and out_path.stat().st_size > 0
