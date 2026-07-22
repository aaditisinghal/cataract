"""Determinism + provenance. Written first, imported everywhere (MASTER_PLAN S1).

Two jobs:
  1. ``seed_everything`` — make a run repeatable across seeds within reported CI. NOTE: this is
     *not* a promise of bitwise-identical GPU runs. ``torch.use_deterministic_algorithms`` is set
     with ``warn_only=True`` because several ops in the diffusion attack have no deterministic CUDA
     kernel and would otherwise raise (see GCP_COMPUTE_PLAN gotcha #4 and RESEARCH_PROTOCOL S7).
  2. ``run_fingerprint`` — stamp every artifact with git sha + dirtiness + framework/GPU versions.
     A dirty tree marks results provisional; enforce "no dirty result in a paper table" downstream.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

# Repo root = two levels up from this file (patchguard/repro.py -> repo/).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (incl. CUDA) and pin deterministic knobs where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs; harmless on CPU-only machines.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a core dep; guard only so repro can't hard-crash imports.
        pass

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Scoped determinism: warn (don't raise) on ops without deterministic kernels.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # very old torch without warn_only
        torch.use_deterministic_algorithms(True)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_fingerprint() -> dict[str, Any]:
    """Provenance stamped into every checkpoint / metrics.json / figure.

    If git state can't be read, ``git_dirty`` is reported ``True`` (conservative: unknown == unsafe),
    so such results are treated as provisional and barred from paper tables downstream.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if sha is None:
        # No git tree (e.g. inside the built container). Fall back to build-stamped provenance
        # written by scripts/10_build_image.sh so container runs are still paper-ready.
        try:
            from patchguard import _buildinfo  # type: ignore

            sha = _buildinfo.GIT_SHA
            git_dirty = bool(getattr(_buildinfo, "GIT_DIRTY", True))
        except Exception:
            git_dirty = True
    else:
        # dirty if there are uncommitted changes, or if git state is unknown.
        git_dirty = True if status is None else bool(status.strip())

    fp: dict[str, Any] = {
        "git_sha": sha or "unknown",
        "git_dirty": git_dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import numpy as np

        fp["numpy"] = np.__version__
    except ImportError:
        fp["numpy"] = None

    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
        fp["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        fp["torch"] = fp["cuda"] = fp["gpu"] = None

    return fp


def is_paper_ready(fingerprint: dict[str, Any]) -> bool:
    """A result may enter a paper table only from a clean tree with a known commit."""
    return not fingerprint.get("git_dirty", True) and fingerprint.get("git_sha") != "unknown"
