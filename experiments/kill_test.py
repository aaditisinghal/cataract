"""The go/no-go gate (MASTER_PLAN ★).

Thresholds are PRE-REGISTERED here (committed to git before the real run). Do not edit them after
seeing real data — git history is the witness.

Design (exactly this, nothing more):
  200 FUNSD docs · ColPali vs BiPali at matched bytes/page · attack v0 only ·
  patch-scoped vs flat Gaussian × 6 noise levels · 5 seeds.

Gate (BOTH must hold):
  1. PFRR delta (ColPali - BiPali) >= 15pp
  2. frontier-AUC(patch-scoped) - AUC(flat) CI excludes 0 at 95% (patch-scoped dominates)

Two run modes:
  --mock   Runs the FULL orchestration + stats on synthetic measurements. No GPU/data/OCR. Proves the
           plumbing and writes a fingerprinted results file. Runnable in CI.
  (real)   Fills the same measurement arrays from the pipeline: encode (ColPali/BiPali) -> defend
           (flat / oracle-patch-scoped Gaussian × noise) -> invert (trained decoder) -> OCR -> PFRR,
           plus retrieval utility. Blocked on: attack/decoder training (S6) + OCR wiring (S4 real) +
           FUNSD download. Wired at the ★ stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from patchguard.eval.killgate import assemble_and_gate
from patchguard.repro import run_fingerprint, seed_everything

# --- PRE-REGISTERED THRESHOLDS (do not edit after the run) --------------------------------
PFRR_DELTA_MIN_PP = 15.0
N_DOCS = 200
N_SEEDS = 5
NOISE_LEVELS = 6
# ------------------------------------------------------------------------------------------

RESULTS_DIR = Path("results/kill_test")


def _mock_measurements(seed: int) -> dict[str, np.ndarray]:
    """Synthetic per-doc measurements shaped exactly like the real pipeline's output.

    Encodes the paper's hypotheses so the plumbing exercises a GO path: ColPali recovers more PII than
    pooled BiPali, and patch-scoped Gaussian holds higher privacy at equal utility than flat.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    n, k = N_DOCS, NOISE_LEVELS
    rec_colpali = (rng.random(n) < 0.64).astype(float)  # undefended PII recovery
    rec_bipali = (rng.random(n) < 0.46).astype(float)
    noise = np.linspace(0.0, 1.0, k)
    util_patch = np.clip(1.0 - 0.10 * noise + 0.01 * rng.standard_normal((n, k)), 0, 1)
    util_flat = np.clip(1.0 - 0.35 * noise + 0.01 * rng.standard_normal((n, k)), 0, 1)
    base_priv = 0.30 + 0.60 * noise
    priv_patch = np.clip(base_priv + 0.15 + 0.01 * rng.standard_normal((n, k)), 0, 1)
    priv_flat = np.clip(base_priv + 0.01 * rng.standard_normal((n, k)), 0, 1)
    return dict(
        recovery_colpali_undef=rec_colpali,
        recovery_bipali_undef=rec_bipali,
        util_patch=util_patch,
        priv_patch=priv_patch,
        util_flat=util_flat,
        priv_flat=priv_flat,
    )


def run_mock() -> dict[str, object]:
    m = _mock_measurements(seed=0)
    result = assemble_and_gate(**m, min_delta_pp=PFRR_DELTA_MIN_PP, n_resamples=2000, seed=0)
    fp = run_fingerprint()
    payload: dict[str, object] = {
        "mode": "mock",
        "config": {
            "n_docs": N_DOCS,
            "n_seeds": N_SEEDS,
            "noise_levels": NOISE_LEVELS,
            "pfrr_delta_min_pp": PFRR_DELTA_MIN_PP,
        },
        "result": {
            "pfrr_colpali": result.pfrr_colpali,
            "pfrr_bipali": result.pfrr_bipali,
            "pfrr_delta_pp": result.pfrr_delta_pp,
            "delta_pass": result.delta_pass,
            "auc_diff": result.auc_diff,
            "auc_ci": list(result.auc_ci),
            "frontier_pass": result.frontier_pass,
            "decision": result.decision,
        },
        "fingerprint": fp,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "metrics_mock.json"
    out.write_text(json.dumps(payload, indent=2))
    print(result.summary())
    print(f"[mock] wrote {out}  (paper_ready={not fp['git_dirty'] and fp['git_sha'] != 'unknown'})")
    return payload


def run_real() -> None:
    raise NotImplementedError(
        "Real kill test is pre-registered but not yet wired. Blocked on: trained attack/decoder (S6), "
        "OCR wiring in eval/pfrr (S4-real), and FUNSD download. Fills the same arrays run_mock() "
        "synthesizes, then calls assemble_and_gate(). See docs/MASTER_PLAN.md ★."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="run full orchestration on synthetic data")
    args = ap.parse_args()
    if args.mock:
        run_mock()
    else:
        run_real()


if __name__ == "__main__":
    main()
