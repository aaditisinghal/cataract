"""Privacy-utility frontier statistics (MASTER_PLAN S4, stats plan RESEARCH_PROTOCOL S4).

Pure numpy + stdlib (no scipy) so it runs in the CPU CI. Everything reports uncertainty: the kill
test's second threshold is "frontier-AUC difference excludes 0 at 95%", which is exactly
``frontier_auc_diff_ci`` below.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[..., np.ndarray] = np.mean,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, lo, hi).

    ``statistic`` must accept ``axis=`` (np.mean/np.median do). Default = mean, i.e. a recovery-rate
    CI when ``values`` is a per-document boolean/0-1 array.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    n = vals.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = statistic(vals[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(statistic(vals)), float(lo), float(hi)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns (z, two_sided_p).

    Use for recovery-rate deltas, e.g. ColPali vs BiPali in the kill test.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1, n2 must be positive")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_two_sided = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return float(z), float(p_two_sided)


def frontier_auc(utility: np.ndarray, privacy: np.ndarray) -> float:
    """Area under the privacy-vs-utility frontier.

    ``privacy`` is a higher-is-better defense metric (e.g. 1 - PFRR). Points are sorted by utility
    and integrated with the trapezoid rule over the utility axis. Higher AUC = better privacy at
    equal utility.
    """
    u = np.asarray(utility, dtype=float)
    p = np.asarray(privacy, dtype=float)
    if u.shape != p.shape or u.ndim != 1 or u.size < 2:
        raise ValueError("utility and privacy must be 1-D of equal length >= 2")
    order = np.argsort(u)
    return float(np.trapz(p[order], u[order]))


def dominance_at(
    utility: np.ndarray,
    privacy_a: np.ndarray,
    privacy_b: np.ndarray,
    levels: tuple[float, ...] = (0.99, 0.97, 0.95),
    undefended_utility: float = 1.0,
) -> dict[float, dict[str, float]]:
    """Compare two defenses' privacy at fixed utility levels (e.g. within 1%/3%/5% of undefended).

    Privacy is linearly interpolated onto each target utility. Returns
    ``{level: {"a": pa, "b": pb, "delta": pa-pb}}`` where delta>0 means A dominates B there.
    """
    u = np.asarray(utility, dtype=float)
    order = np.argsort(u)
    u_s = u[order]
    a_s = np.asarray(privacy_a, dtype=float)[order]
    b_s = np.asarray(privacy_b, dtype=float)[order]
    out: dict[float, dict[str, float]] = {}
    for lvl in levels:
        target = lvl * undefended_utility
        pa = float(np.interp(target, u_s, a_s))
        pb = float(np.interp(target, u_s, b_s))
        out[lvl] = {"a": pa, "b": pb, "delta": pa - pb}
    return out


def frontier_auc_diff_ci(
    utility: np.ndarray,
    privacy_a_per_doc: np.ndarray,
    privacy_b_per_doc: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for AUC(A) - AUC(B), resampling documents (paired).

    ``privacy_*_per_doc`` are (n_docs, n_points) arrays of the higher-is-better privacy metric at
    each shared utility point; ``utility`` is (n_points,). The kill-test gate passes when the
    returned CI excludes 0. Returns (point_diff, lo, hi).
    """
    u = np.asarray(utility, dtype=float)
    a = np.asarray(privacy_a_per_doc, dtype=float)
    b = np.asarray(privacy_b_per_doc, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != u.size:
        raise ValueError("per-doc arrays must be (n_docs, n_points) matching utility length")
    n_docs = a.shape[0]
    rng = np.random.default_rng(seed)

    def _auc_diff(a_docs: np.ndarray, b_docs: np.ndarray) -> float:
        return frontier_auc(u, a_docs.mean(axis=0)) - frontier_auc(u, b_docs.mean(axis=0))

    point = _auc_diff(a, b)
    boot = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        idx = rng.integers(0, n_docs, size=n_docs)
        boot[r] = _auc_diff(a[idx], b[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
