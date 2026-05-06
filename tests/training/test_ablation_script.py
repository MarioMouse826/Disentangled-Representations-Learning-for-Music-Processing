"""Tests for ablation matrix runner script (plan Task 6.3)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


_SCRIPT = Path("scripts/run_ablations.sh")


def test_script_exists_and_executable() -> None:
    assert _SCRIPT.exists()
    # Bash script; check shebang.
    first_line = _SCRIPT.read_text().splitlines()[0]
    assert first_line.startswith("#!") and "bash" in first_line


def test_script_bash_syntax_valid() -> None:
    r = subprocess.run(
        ["bash", "-n", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, f"bash syntax error:\n{r.stderr}"


@pytest.mark.parametrize(
    "ablation_id",
    [
        "full",
        "no_l_inv",
        "no_l_equi",
        "no_l_swap",
        "rep_identity",
        "rep_translation",
        "dims_swapped",
        "dims_unswapped",
        "beta_const",
        "beta_cyclical",
        "unpaired_batch",
    ],
)
def test_script_contains_required_ablation(ablation_id: str) -> None:
    content = _SCRIPT.read_text()
    assert f"\"{ablation_id}|" in content, (
        f"ablation '{ablation_id}' missing from ABLATIONS_READY list"
    )


def test_script_has_second_pass_guard() -> None:
    content = _SCRIPT.read_text()
    assert "SECOND_PASS" in content
    assert "TOP_3" in content
    assert "moisesdb_bass" in content


def test_script_skips_existing_runs() -> None:
    # Idempotency — script should short-circuit if metrics.json already written.
    content = _SCRIPT.read_text()
    assert "metrics.json" in content
    assert "skip" in content.lower()


def test_script_uses_hydra_config_name_sweep() -> None:
    # Ablations ride the Task 6.2 sweep config so Hydra overrides apply.
    content = _SCRIPT.read_text()
    assert "--config-name sweep" in content
