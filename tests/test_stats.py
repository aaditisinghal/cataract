"""Unit tests for patchguard.eval.stats (plan A2). Pure CPU / numpy."""

import numpy as np

from patchguard.eval.stats import (
    bootstrap_ci,
    bootstrap_pvalue,
    holm_bonferroni,
    mcnemar_exact,
    ndcg_at_k,
    two_proportion_ztest,
)


# --- bootstrap_ci --------------------------------------------------------------------------------
def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.random(400)
    est, lo, hi = bootstrap_ci(x, iters=2000, seed=1)
    assert lo < est < hi
    assert abs(est - x.mean()) < 1e-9  # point estimate is the sample mean


def test_bootstrap_ci_constant_is_degenerate():
    est, lo, hi = bootstrap_ci(np.full(20, 0.5), iters=500, seed=0)  # 0.5 exactly representable
    assert est == lo == hi == 0.5


# --- two_proportion_ztest ------------------------------------------------------------------------
def test_two_proportion_ztest_significant_gap():
    z, p = two_proportion_ztest(k1=107, n1=165, k2=82, n2=165)  # ~0.65 vs ~0.50
    assert z > 0 and p < 0.05


def test_two_proportion_ztest_no_gap():
    z, p = two_proportion_ztest(k1=50, n1=100, k2=50, n2=100)
    assert abs(z) < 1e-9 and p > 0.99


# --- mcnemar_exact -------------------------------------------------------------------------------
def test_mcnemar_all_discordance_one_way_is_significant():
    # 10 items flipped one way, none the other => p = 2 * 0.5**10 = 1/512.
    p = mcnemar_exact(b=10, c=0)
    assert p < 0.05
    assert abs(p - 2.0 * 0.5**10) < 1e-12


def test_mcnemar_balanced_is_not_significant():
    p = mcnemar_exact(b=5, c=5)
    assert p == 1.0  # symmetric, clipped to 1


def test_mcnemar_no_discordant_pairs_is_one():
    assert mcnemar_exact(b=0, c=0) == 1.0


def test_mcnemar_more_imbalance_lowers_p():
    # Same total n, more lopsided split => smaller p.
    assert mcnemar_exact(b=12, c=0) < mcnemar_exact(b=8, c=4)


# --- bootstrap_pvalue ----------------------------------------------------------------------------
def test_bootstrap_pvalue_far_from_null_is_small():
    x = np.array([0.60, 0.63, 0.58, 0.61, 0.59])  # clearly > 0
    assert bootstrap_pvalue(x, null=0.0, iters=4000, seed=0) < 0.05


def test_bootstrap_pvalue_centered_on_null_is_large():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=200)
    assert bootstrap_pvalue(x, null=0.0, iters=4000, seed=1) > 0.2


# --- holm_bonferroni -----------------------------------------------------------------------------
def test_holm_ordering_monotonic_and_bounded():
    pv = {"a": 0.001, "b": 0.02, "c": 0.5, "d": 0.04}
    res = holm_bonferroni(pv, alpha=0.05)
    # adjusted p is monotone non-decreasing when walking the raw p-values in ascending order
    for name in pv:
        assert res[name]["p_adj"] >= res[name]["p_raw"] - 1e-12  # adjustment never lowers p
        assert 0.0 <= res[name]["p_adj"] <= 1.0
    order = sorted(pv, key=lambda k: pv[k])
    adj_in_order = [res[k]["p_adj"] for k in order]
    assert all(adj_in_order[i] <= adj_in_order[i + 1] + 1e-12 for i in range(len(adj_in_order) - 1))


def test_holm_rejects_smallest_only_when_appropriate():
    pv = {"a": 0.001, "b": 0.02, "c": 0.5, "d": 0.04}
    res = holm_bonferroni(pv, alpha=0.05)
    # m=4: a -> 4*0.001=0.004 (reject); b -> 3*0.02=0.06 (>0.05 stop) => only a rejected.
    assert res["a"]["reject"] is True
    assert abs(res["a"]["p_adj"] - 0.004) < 1e-12
    assert res["b"]["reject"] is False
    assert res["c"]["reject"] is False and res["d"]["reject"] is False


def test_holm_all_tiny_all_reject():
    res = holm_bonferroni({"x": 1e-4, "y": 2e-4, "z": 3e-4}, alpha=0.05)
    assert all(v["reject"] for v in res.values())


def test_holm_empty_input():
    assert holm_bonferroni({}) == {}


# --- ndcg_at_k -----------------------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    assert abs(ndcg_at_k([3, 2, 1, 0], k=4) - 1.0) < 1e-12


def test_ndcg_worst_ranking_below_perfect():
    ascending = ndcg_at_k([0, 1, 2, 3], k=4)  # relevant items buried at the bottom
    assert 0.0 < ascending < 1.0


def test_ndcg_all_zero_relevance_is_zero():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_truncation_uses_top_k():
    # A relevant item beyond k should not count.
    full = ndcg_at_k([0, 0, 3], k=2)  # top-2 has no relevance
    assert full == 0.0


def test_ndcg_matches_hand_computation():
    # rel = [3, 2, 3] ; DCG = 3/log2(2) + 2/log2(3) + 3/log2(4)
    rel = [3, 2, 3]
    dcg = 3 / np.log2(2) + 2 / np.log2(3) + 3 / np.log2(4)
    ideal = sorted(rel, reverse=True)  # [3,3,2]
    idcg = ideal[0] / np.log2(2) + ideal[1] / np.log2(3) + ideal[2] / np.log2(4)
    assert abs(ndcg_at_k(rel, k=3) - dcg / idcg) < 1e-12
