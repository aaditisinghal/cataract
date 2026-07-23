"""Paper-grade significance + ranking statistics (plan A2).

Pure numpy + stdlib (no scipy) so every test in this file runs in the CPU CI. This module is the
one place the paper's claims turn into numbers with uncertainty and multiplicity control:

  * ``bootstrap_ci``          — mean ± 95% percentile-bootstrap CI (the interval on every headline
                                recovery rate / defense delta). Thin wrapper over
                                ``frontier.bootstrap_ci`` so there is exactly ONE bootstrap kernel.
  * ``two_proportion_ztest``  — recovery-rate delta between two arms (alias of
                                ``frontier.two_proportion_z``; re-exported to avoid duplication).
  * ``mcnemar_exact``         — PAIRED test for two attacks/defenses on the SAME documents (uses the
                                discordant pair counts b, c; the correct test when the same cards are
                                scored twice, e.g. ordered vs shuffled, ColPali vs BiPali on a shared
                                page set).
  * ``bootstrap_pvalue``      — two-sided bootstrap p-value that a per-seed mean differs from a null
                                (the p-values ``aggregate_seeds`` feeds into Holm across a seed sweep).
  * ``holm_bonferroni``       — step-down family-wise error control over the primary claim family
                                (name/id/dob recovery deltas, defense-dominance deltas). Guards against
                                "one of six comparisons came out p<0.05" false positives.
  * ``ndcg_at_k``             — graded ranking quality of the dictionary attack's candidate list
                                (a soft complement to top-1 accuracy).

A positive claim in the paper is one whose bootstrap CI excludes the null AND whose Holm-adjusted
p-value stays below 0.05; this module computes both halves of that gate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Reuse the single bootstrap/z-test kernel from frontier.py — do NOT reimplement it here.
from patchguard.eval.frontier import bootstrap_ci as _frontier_bootstrap_ci
from patchguard.eval.frontier import two_proportion_z as _frontier_two_proportion_z


def bootstrap_ci(
    samples: np.ndarray,
    iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile-bootstrap mean CI. Returns (mean, lo, hi).

    Wrapper over ``frontier.bootstrap_ci`` with the plan's argument names (``iters`` == n_resamples).
    ``samples`` is a 1-D array — e.g. one recovery-rate per seed, or one 0/1 hit per document.
    """
    return _frontier_bootstrap_ci(
        np.asarray(samples, dtype=float), n_resamples=iters, alpha=alpha, seed=seed
    )


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z, two_sided_p).

    Re-exports ``frontier.two_proportion_z`` under the plan's name. Use for UNPAIRED recovery-rate
    deltas (two independent corpora / arms).
    """
    return _frontier_two_proportion_z(k1, n1, k2, n2)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) two-sided McNemar test on discordant pair counts. Returns p.

    For two binary classifiers/attacks scored on the SAME items, ``b`` = # items the first got right
    and the second wrong, ``c`` = # the first wrong and second right. Concordant pairs carry no
    information and are ignored. Under H0 (equal accuracy) each discordant pair is a fair coin, so
    with ``n = b + c`` and ``k = min(b, c)``::

        p = min(1, 2 * sum_{i=0}^{k} C(n, i) * 0.5**n)

    ``b == c == 0`` (no discordant pairs -> no evidence) returns p = 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative discordant counts")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def bootstrap_pvalue(
    samples: np.ndarray,
    null: float = 0.0,
    iters: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value that E[samples] == ``null``. Returns p in [0, 1].

    Resamples the mean and reports ``2 * min(P(mean <= null), P(mean >= null))`` (clipped to 1).
    This is the per-metric significance screen ``aggregate_seeds`` computes from a small seed sweep
    before Holm-correcting across the claim family. Degenerate (all samples equal ``null``) -> 1.0.
    """
    vals = np.asarray(samples, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("samples must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    n = vals.size
    idx = rng.integers(0, n, size=(iters, n))
    boot = vals[idx].mean(axis=1)
    p_le = float(np.mean(boot <= null))
    p_ge = float(np.mean(boot >= null))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def holm_bonferroni(
    pvalues: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down FWER control.

    Input maps a comparison name -> its raw p-value. Returns, per name,
    ``{"p_raw": float, "p_adj": float, "reject": bool}`` where ``p_adj`` is the monotone step-down
    adjusted p-value (rank i in ascending order gets ``(m - i) * p``, then a running max enforces
    monotonicity, then clip to 1) and ``reject`` is ``p_adj <= alpha``. Holm strongly controls the
    family-wise error rate and is uniformly more powerful than plain Bonferroni.
    """
    if not pvalues:
        return {}
    names = list(pvalues.keys())
    raw = np.array([float(pvalues[n]) for n in names], dtype=float)
    if np.any((raw < 0) | (raw > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = raw.size
    order = np.argsort(raw, kind="stable")  # ascending
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * raw[idx]  # Holm multiplier for this rank
        running = max(running, val)  # enforce monotone non-decreasing in sorted order
        adj_sorted[rank] = min(1.0, running)
    out: dict[str, dict[str, Any]] = {}
    for rank, idx in enumerate(order):
        out[names[idx]] = {
            "p_raw": float(raw[idx]),
            "p_adj": float(adj_sorted[rank]),
            "reject": bool(adj_sorted[rank] <= alpha),
        }
    return out


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k. Returns nDCG in [0, 1].

    ``relevances`` are the graded relevances of items in the order the system ranked them
    (index 0 = top result). DCG@k = sum_i rel_i / log2(i + 2); the ideal DCG uses the same
    relevances sorted descending. All-zero relevance (nothing relevant) returns 0.0.
    """
    rel = np.asarray(relevances, dtype=float).ravel()
    if k <= 0:
        raise ValueError("k must be positive")
    kk = min(k, rel.size)
    if kk == 0:
        return 0.0

    def _dcg(scores: np.ndarray) -> float:
        discounts = 1.0 / np.log2(np.arange(2, scores.size + 2))
        return float(np.sum(scores * discounts))

    dcg = _dcg(rel[:kk])
    ideal = np.sort(rel)[::-1][:kk]
    idcg = _dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)
