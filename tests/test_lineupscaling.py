"""CPU tests for experiments.lineup_scaling (mock retriever + tiny synthetic cards, no GPU/ColPali).

The scaling + absent-target core is backend-agnostic, so MockRetriever on tiny cards exercises the exact
functions the ColPali run uses. We assert the structural invariants that must hold for ANY retriever:
graceful (monotone) decay of recovery in K, correct lift/chance bookkeeping, and a well-formed ROC.
"""

import numpy as np

from experiments.lineup_scaling import (
    absent_target_eval,
    generate_name_candidates,
    scaling_curve,
    score_pool_per_card,
)
from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
from patchguard.retrievers.mock import MockRetriever


def _mock_setup(n=8, npool=40, seed=0):
    rng = np.random.default_rng(seed)
    retriever = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    cards = generate_id_cards(n, seed=seed, value_font_size=24, vary=True)
    pool = generate_name_candidates(npool, _FIRST, _LAST, rng)
    per_card = score_pool_per_card(retriever, cards, pool, field="name")
    return per_card, pool, cards


def test_candidate_pool_unique_and_sized():
    rng = np.random.default_rng(1)
    # beyond the 240 base -> must generate middle-initial variants and stay unique
    pool = generate_name_candidates(500, _FIRST, _LAST, rng)
    assert len(pool) == 500
    assert len(set(pool)) == 500
    assert all(len(s.split()) >= 2 for s in pool)  # every candidate is name-like


def test_candidate_pool_subsamples_base():
    rng = np.random.default_rng(2)
    pool = generate_name_candidates(50, _FIRST, _LAST, rng)
    assert len(pool) == 50 == len(set(pool))
    base = {f"{a} {b}" for a in _FIRST for b in _LAST}
    assert all(s in base for s in pool)  # small K stays within the closed FIRST x LAST vocab


def test_score_pool_structure_excludes_true():
    per_card, pool, cards = _mock_setup(n=6, npool=30)
    assert len(per_card) == 6
    for r in per_card:
        assert r["true"] in {f"{a} {b}" for a in _FIRST for b in _LAST}
        # distractors = pool with the card's own true value removed (if it was sampled into the pool)
        expected = len(pool) - (1 if r["true"] in pool else 0)
        assert r["dist_scores"].shape[0] == expected
        assert np.isfinite(r["true_score"])


def test_scaling_curve_monotone_decay_and_lift():
    per_card, pool, _ = _mock_setup(n=8, npool=40)
    Ks = [2, 5, 10, 20, 40]
    curve = scaling_curve(per_card, Ks)
    assert [c["K"] for c in curve] == Ks
    recs = [c["recovery"] for c in curve]
    # graceful decay: recovery is monotone non-increasing as the lineup grows (structural)
    assert all(recs[i] >= recs[i + 1] - 1e-12 for i in range(len(recs) - 1))
    for c in curve:
        assert 0.0 <= c["recovery"] <= 1.0
        assert abs(c["chance"] - 1.0 / c["K"]) < 1e-12
        assert abs(c["lift"] - (c["recovery"] / c["chance"])) < 1e-9


def test_scaling_k1_distractorless_is_trivial():
    per_card, _, _ = _mock_setup(n=5, npool=20)
    curve = scaling_curve(per_card, [1])
    # K=1 -> no distractors -> true value is trivially top-1 for every card
    assert curve[0]["recovery"] == 1.0
    assert curve[0]["chance"] == 1.0


def test_absent_target_roc_wellformed():
    per_card, _, _ = _mock_setup(n=12, npool=40)
    res = absent_target_eval(per_card, lineup_size=8, absent_frac=0.5, seed=0)
    # balanced split honored exactly
    assert res["n_absent"] == 6
    assert res["n_present"] == 6
    assert res["lineup_size"] == 8
    assert 0.0 <= res["false_accept"] <= 1.0
    assert 0.0 <= res["true_accept"] <= 1.0
    assert res["auc"] is None or 0.0 <= res["auc"] <= 1.0
    # ROC point list is well-formed
    assert len(res["roc"]) >= 2
    for p in res["roc"]:
        assert 0.0 <= p["fpr"] <= 1.0
        assert 0.0 <= p["tpr"] <= 1.0


def test_absent_target_degenerate_fractions():
    per_card, _, _ = _mock_setup(n=6, npool=30)
    all_present = absent_target_eval(per_card, lineup_size=6, absent_frac=0.0, seed=0)
    assert all_present["n_absent"] == 0
    assert all_present["auc"] is None  # no negative class -> ROC undefined, handled gracefully
    all_absent = absent_target_eval(per_card, lineup_size=6, absent_frac=1.0, seed=0)
    assert all_absent["n_present"] == 0
    assert all_absent["auc"] is None
