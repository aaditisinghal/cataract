"""CPU tests for the utility (B2) and defense-floor (B3) experiments.

Everything runs on the deterministic MockRetriever with tiny tensors and tiny epochs, so no ColPali,
GPU, or network is needed. We only assert the SHAPES and RANGES of the reported quantities (all rates
must be genuine probabilities in [0,1]) and that the sweeps return well-formed frontier lists — the
scientific magnitudes are only meaningful under the real encoder on the GPU.
"""

import numpy as np

from patchguard.retrievers.mock import MockRetriever

from experiments.vidore_utility import compute_retrieval_metrics, run_synthetic
from experiments.defense_floor import run_floor, _blend_query


def _in_unit(x):
    return isinstance(x, float) and 0.0 - 1e-9 <= x <= 1.0 + 1e-9


def test_compute_retrieval_metrics_ranges_and_perfect_case():
    rng = np.random.default_rng(0)
    d = 8
    docs = [rng.standard_normal((5, d)).astype(np.float32) for _ in range(6)]
    docs = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in docs]
    # each query == one of that doc's own patches -> gold must rank #1 (perfect retrieval).
    queries = [docs[i][:1] for i in range(len(docs))]
    golds = list(range(len(docs)))
    m = compute_retrieval_metrics(queries, docs, golds)
    for k in ("ndcg@5", "recall@1", "mrr"):
        assert _in_unit(m[k]), (k, m[k])
    assert m["recall@1"] == 1.0 and m["mrr"] == 1.0


def test_blend_query_endpoints():
    topic = np.ones((2, 4), dtype=np.float32)
    name = 2.0 * np.ones((3, 4), dtype=np.float32)
    b0 = _blend_query(topic, name, 0.0)
    b1 = _blend_query(topic, name, 1.0)
    assert b0.shape == (5, 4) and b1.shape == (5, 4)
    # alpha=0 zeroes the name tokens; alpha=1 zeroes the topic tokens.
    assert np.allclose(b0[2:], 0.0) and np.allclose(b0[:2], 1.0)
    assert np.allclose(b1[:2], 0.0) and np.allclose(b1[2:], 2.0)


def test_run_synthetic_metrics_in_range():
    ret = MockRetriever(dim=8)
    payload = run_synthetic(ret, n_corpus=5, n_queries=5, n_train=6, lam=5.0, epochs=4,
                            seed=0, font_size=24, dim=8, np_train=8, distractors_priv=6,
                            sigmas=(0.2, 0.5))
    assert payload["mode"] == "vidore_utility" and payload["source"] == "synthetic"
    for block in (payload["vanilla"], payload["P"]["metrics"], payload["flat_at_matched_privacy"]):
        for k in ("ndcg@5", "recall@1", "mrr"):
            assert _in_unit(block[k]), (k, block[k])
    assert _in_unit(payload["P"]["privacy"])
    assert isinstance(payload["flat_frontier"], list) and payload["flat_frontier"]
    for f in payload["flat_frontier"]:
        assert _in_unit(f["privacy"])
        for k in ("ndcg@5", "recall@1", "mrr"):
            assert _in_unit(f["metrics"][k])


def test_run_floor_returns_monotone_ish_frontier():
    ret = MockRetriever(dim=8)
    alphas = [0.0, 0.5, 1.0]
    lams = (0.0, 5.0)
    payload = run_floor(ret, alphas=alphas, n_train=6, n_test=5, epochs=4, lams=lams,
                        seed=0, font_size=24, dim=8, priv_target=0.8, np_train=8,
                        distractors_priv=6)
    assert payload["mode"] == "defense_floor"
    assert len(payload["per_alpha"]) == len(alphas)
    for pa, a in zip(payload["per_alpha"], alphas):
        assert pa["alpha"] == a
        assert isinstance(pa["frontier"], list) and len(pa["frontier"]) == len(lams)
        assert _in_unit(pa["util_at_priv_target"])
        for f in pa["frontier"]:
            assert _in_unit(f["privacy"]) and _in_unit(f["utility"])
    # crossover is either None or one of the swept alphas.
    assert payload["crossover_alpha"] is None or payload["crossover_alpha"] in alphas
