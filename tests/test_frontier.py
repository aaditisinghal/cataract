import numpy as np

from patchguard.eval.frontier import (
    bootstrap_ci,
    dominance_at,
    frontier_auc,
    frontier_auc_diff_ci,
    two_proportion_z,
)


def test_bootstrap_ci_of_constant_is_that_constant():
    est, lo, hi = bootstrap_ci(np.ones(50), n_resamples=1000, seed=0)
    assert est == 1.0 and lo == 1.0 and hi == 1.0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    est, lo, hi = bootstrap_ci(x, n_resamples=2000, seed=1)
    assert lo < est < hi
    assert abs(est - x.mean()) < 1e-9


def test_two_proportion_z_significant_gap():
    # 0.50 vs 0.65 at n=165/arm (the powered kill-test design) should be significant.
    z, p = two_proportion_z(k1=107, n1=165, k2=82, n2=165)  # ~0.65 vs ~0.50
    assert z > 0
    assert p < 0.05


def test_two_proportion_z_no_gap():
    z, p = two_proportion_z(k1=50, n1=100, k2=50, n2=100)
    assert abs(z) < 1e-9
    assert p > 0.99


def test_frontier_auc_higher_privacy_curve_has_higher_auc():
    u = np.array([0.90, 0.93, 0.96, 0.99])
    low = np.array([0.2, 0.25, 0.3, 0.35])
    high = np.array([0.6, 0.65, 0.7, 0.75])
    assert frontier_auc(u, high) > frontier_auc(u, low)


def test_frontier_auc_unsorted_input_ok():
    u = np.array([0.99, 0.90, 0.96, 0.93])
    p = np.array([0.75, 0.60, 0.70, 0.65])
    # equals the sorted computation
    order = np.argsort(u)
    assert np.isclose(frontier_auc(u, p), np.trapz(p[order], u[order]))


def test_dominance_at_reports_positive_delta_when_a_better():
    u = np.array([0.90, 0.95, 1.00])
    a = np.array([0.7, 0.6, 0.5])
    b = np.array([0.4, 0.3, 0.2])
    dom = dominance_at(u, a, b, levels=(0.99, 0.95), undefended_utility=1.0)
    assert dom[0.95]["delta"] > 0
    assert np.isclose(dom[0.95]["a"], 0.6)


def test_auc_diff_ci_excludes_zero_when_clearly_separated():
    # Defense A dominates B on every doc => CI for AUC(A)-AUC(B) should exclude 0.
    u = np.array([0.90, 0.95, 1.00])
    rng = np.random.default_rng(0)
    a = 0.7 + 0.02 * rng.standard_normal((60, 3))
    b = 0.3 + 0.02 * rng.standard_normal((60, 3))
    point, lo, hi = frontier_auc_diff_ci(u, a, b, n_resamples=1000, seed=2)
    assert point > 0
    assert lo > 0  # excludes zero => kill-test frontier threshold would pass
