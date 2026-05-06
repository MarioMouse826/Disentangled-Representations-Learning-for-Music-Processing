"""Run manifest writer — captures per-run provenance for reproducibility.

Collects git SHA, config dump, pip freeze, PyTorch/CUDA versions, hardware
info, and a UTC timestamp. Writes JSON suitable for long-term archival
alongside checkpoints + logs.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _git_commit() -> str | None:
    """Current HEAD SHA, or None if not in a git repo / git not installed."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        return sha if len(sha) == 40 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool | None:
    """True if working tree has uncommitted changes; None if unknown."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _pip_freeze() -> list[str]:
    """List of `pkg==ver` strings; empty list on failure."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if out.returncode != 0:
            return []
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _cuda_version() -> str | None:
    return torch.version.cuda if torch.cuda.is_available() else None


def _cudnn_version() -> int | None:
    if not torch.backends.cudnn.is_available():
        return None
    try:
        return int(torch.backends.cudnn.version() or 0) or None
    except Exception:
        return None


def _gpu_info() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    info: list[dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return info


def build_manifest(
    *,
    config: dict[str, Any],
    seed: int,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest dict. JSON-serializable by construction.

    Args:
        budget: optional compute-budget summary from
            `src.utils.compute_budget.BudgetTracker.summary()` — keys
            `wall_time_hours`, `gpu_hours`, `peak_vram_gb`, and optionally
            `flops_per_step`. Included under the `"compute_budget"` field.
    """
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "seed": int(seed),
        "config": config,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": _cuda_version(),
        "cudnn_version": _cudnn_version(),
        "gpu_info": _gpu_info(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pip_freeze": _pip_freeze(),
        "compute_budget": budget or {},
    }


def write_manifest(
    *,
    config: dict[str, Any],
    seed: int,
    out_path: Path | str,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest and write it to `out_path` as JSON."""
    manifest = build_manifest(config=config, seed=seed, budget=budget)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
