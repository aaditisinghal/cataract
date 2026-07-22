"""The ★ kill-test decision (MASTER_PLAN gate #2). Pure + tested; thresholds pre-registered.

Given the two headline measurements, return a branch:
  * both thresholds pass                      -> "GO"           (title earned, execute S6-S11)
  * PFRR delta passes, frontier does not      -> "MEASUREMENT"  (retitle, demote defense, NeurIPS D&B)
  * PFRR delta fails                          -> "STOP"         (architecture story wrong; reassess)

Thresholds live in experiments/kill_test.py (committed before the run). This module only applies them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GO = "GO"
MEASUREMENT = "MEASUREMENT"
STOP = "STOP"


@dataclass(frozen=True)
class KillGateResult:
    pfrr_colpali: float
    pfrr_bipali: float
    pfrr_delta_pp: float
    delta_pass: bool
    auc_diff: float
    auc_ci: tuple[float, float]
    frontier_pass: bool
    decision: str

    def summary(self) -> str:
        d = "PASS" if self.delta_pass else "FAIL"
        f = "PASS" if self.frontier_pass else "FAIL"
        return (
            f"PFRR delta = {self.pfrr_delta_pp:+.1f}pp [{d}]  |  "
            f"AUC(patch-scoped)-AUC(flat) = {self.auc_diff:+.4f} "
            f"CI[{self.auc_ci[0]:+.4f},{self.auc_ci[1]:+.4f}] [{f}]  ->  {self.decision}"
        )


def kill_gate(
    pfrr_colpali: float,
    pfrr_bipali: float,
    auc_diff_point: float,
    auc_ci_lo: float,
    auc_ci_hi: float,
    min_delta_pp: float = 15.0,
) -> KillGateResult:
    """Apply the two pre-registered thresholds and return the branch.

    ``pfrr_*`` are recovery rates in [0,1]. Frontier passes only when patch-scoped STRICTLY dominates
    (CI for AUC(patch-scoped) - AUC(flat) lies entirely above 0).
    """
    delta_pp = (pfrr_colpali - pfrr_bipali) * 100.0
    delta_pass = delta_pp >= min_delta_pp
    frontier_pass = auc_ci_lo > 0.0  # entire CI above zero => patch-scoped dominates

    if delta_pass and frontier_pass:
        decision = GO
    elif delta_pass:
        decision = MEASUREMENT
    else:
        decision = STOP

    return KillGateResult(
        pfrr_colpali=pfrr_colpali,
        pfrr_bipali=pfrr_bipali,
        pfrr_delta_pp=delta_pp,
        delta_pass=delta_pass,
        auc_diff=auc_diff_point,
        auc_ci=(auc_ci_lo, auc_ci_hi),
        frontier_pass=frontier_pass,
        decision=decision,
    )


def _sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def _paired_auc_diff(
    util_a: np.ndarray, priv_a: np.ndarray, util_b: np.ndarray, priv_b: np.ndarray, grid_n: int = 64
) -> float:
    """AUC(A) - AUC(B) computed over a SHARED utility grid (the overlap of the two ranges).

    Comparing each curve over its own utility range is wrong: a defense that degrades utility more
    spans a wider axis and accrues more area even while being strictly worse at equal utility. So we
    interpolate both curves' privacy onto a common utility grid and integrate there — the frontier is
    "privacy at equal utility", and higher = dominates.
    """
    ua, pa = _sorted_xy(util_a.mean(axis=0), priv_a.mean(axis=0))
    ub, pb = _sorted_xy(util_b.mean(axis=0), priv_b.mean(axis=0))
    lo = max(ua.min(), ub.min())
    hi = min(ua.max(), ub.max())
    if hi <= lo:
        return 0.0  # curves don't share a utility range
    grid = np.linspace(lo, hi, grid_n)
    pa_i = np.interp(grid, ua, pa)
    pb_i = np.interp(grid, ub, pb)
    return float(np.trapz(pa_i, grid) - np.trapz(pb_i, grid))


def assemble_and_gate(
    recovery_colpali_undef: np.ndarray,  # (n_docs,) 0/1, ColPali, sigma=0
    recovery_bipali_undef: np.ndarray,  # (n_docs,) 0/1, BiPali, sigma=0
    util_patch: np.ndarray,  # (n_docs, n_noise) retrieval utility, patch-scoped defense
    priv_patch: np.ndarray,  # (n_docs, n_noise) privacy = 1 - PII recovery, patch-scoped
    util_flat: np.ndarray,  # (n_docs, n_noise) utility, flat gaussian
    priv_flat: np.ndarray,  # (n_docs, n_noise) privacy, flat gaussian
    min_delta_pp: float = 15.0,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> KillGateResult:
    """Turn per-document kill-test measurements into the gate decision.

    Bootstraps documents (paired) for the AUC-difference CI so the frontier threshold is exactly
    "patch-scoped dominates flat at 95%". PFRR delta is ColPali vs BiPali on the undefended pages.
    """
    arrays = [util_patch, priv_patch, util_flat, priv_flat]
    n_docs = util_patch.shape[0]
    if any(a.shape[0] != n_docs for a in arrays) or util_patch.shape != priv_patch.shape:
        raise ValueError("per-doc measurement arrays must share (n_docs, n_noise)")

    point = _paired_auc_diff(util_patch, priv_patch, util_flat, priv_flat)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        idx = rng.integers(0, n_docs, size=n_docs)
        boot[r] = _paired_auc_diff(
            util_patch[idx], priv_patch[idx], util_flat[idx], priv_flat[idx]
        )
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    return kill_gate(
        pfrr_colpali=float(recovery_colpali_undef.mean()),
        pfrr_bipali=float(recovery_bipali_undef.mean()),
        auc_diff_point=float(point),
        auc_ci_lo=float(lo),
        auc_ci_hi=float(hi),
        min_delta_pp=min_delta_pp,
    )
