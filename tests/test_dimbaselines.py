"""CPU tests for the dimensionality-reduction baselines (experiments/dim_baselines.py).

Everything runs on the MOCK retriever + tiny tensors: no colpali/torch-model download. Protects:
  (1) both reducers (random projection, PCA) output unit-normalized patches of the reduced dimension;
  (2) reducers are applied CONSISTENTLY to index + query so MaxSim scores in the reduced space;
  (3) the frontier point is well-formed (utility, privacy in [0,1]) end-to-end on the mock retriever;
  (4) the Cataract-match predicate behaves at its threshold.
"""

import numpy as np
import pytest

from experiments.dim_baselines import (
    CATARACT_REF,
    dimreduce_frontier_point,
    fit_pca,
    make_pca_proj,
    make_random_proj,
    matches_cataract,
    random_projection_matrix,
)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _rand_patches(seed=0, npch=24, d=8):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal((npch, d)).astype(np.float32))


def test_random_projection_reduces_dim_and_normalizes():
    x = _rand_patches(seed=1, npch=20, d=8)
    R = random_projection_matrix(8, 4, np.random.default_rng(0))
    assert R.shape == (8, 4)
    y = make_random_proj(R)(x)
    assert y.shape == (20, 4)
    assert np.allclose(np.linalg.norm(y, axis=-1), 1.0, atol=1e-4)


def test_random_projection_clamps_dout_to_din():
    R = random_projection_matrix(8, 32, np.random.default_rng(0))  # d_out > d_in
    assert R.shape == (8, 8)


def test_pca_projection_reduces_dim_and_normalizes():
    x = _rand_patches(seed=2, npch=40, d=8)
    mean, comps = fit_pca(x, 4)
    assert comps.shape == (4, 8)
    y = make_pca_proj(mean, comps)(x)
    assert y.shape == (40, 4)
    assert np.allclose(np.linalg.norm(y, axis=-1), 1.0, atol=1e-4)


def test_pca_components_are_orthonormal():
    x = _rand_patches(seed=3, npch=50, d=8)
    _, comps = fit_pca(x, 4)
    gram = comps @ comps.T
    assert np.allclose(gram, np.eye(4), atol=1e-4)


def test_pca_rank_clamped_when_dout_exceeds_available():
    # Only 3 rows -> at most rank-3 (minus the mean) worth of components available.
    x = _rand_patches(seed=4, npch=3, d=8)
    _, comps = fit_pca(x, 6)
    assert comps.shape[0] <= 3
    assert comps.shape[1] == 8


def test_reducer_applied_to_query_too_keeps_dims_consistent():
    # A reducer must map query tokens into the SAME reduced space as the stored patches.
    q = _rand_patches(seed=5, npch=5, d=8)
    R = random_projection_matrix(8, 4, np.random.default_rng(0))
    proj = make_random_proj(R)
    assert proj(q).shape == (5, 4)  # query reduced identically to index


def test_matches_cataract_predicate_threshold():
    # A point at least as good as the reference on both axes matches; a clearly worse one does not.
    good = {"privacy": CATARACT_REF["privacy"], "utility": CATARACT_REF["utility"]}
    bad = {"privacy": 0.2, "utility": 0.2}
    assert matches_cataract(good) is True
    assert matches_cataract(bad) is False


def test_frontier_point_end_to_end_on_mock():
    from experiments.baseline_frontier import gen_cards, make_qcache
    from patchguard.data.synthdoc import _FIRST, _LAST
    from patchguard.retrievers.mock import MockRetriever

    retriever = MockRetriever(grid=(4, 4), dim=8, seed=0)
    q = make_qcache(retriever)
    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    rng = np.random.default_rng(0)
    names = pool[:20]
    test = gen_cards(retriever, names, k=4, seed0=100, font_size=20, rng=rng)
    for c in test:
        q(c["topic"])

    # random projection 8 -> 4
    R = random_projection_matrix(8, 4, rng)
    u, p = dimreduce_frontier_point(test, make_random_proj(R), q, pool, distractors=5, rng=rng)
    assert 0.0 <= u <= 1.0
    assert 0.0 <= p <= 1.0

    # PCA 8 -> 4 fit on the same cards' patches
    stack = np.concatenate([c["patches"] for c in test], axis=0)
    mean, comps = fit_pca(stack, 4)
    u2, p2 = dimreduce_frontier_point(test, make_pca_proj(mean, comps), q, pool, distractors=5, rng=rng)
    assert 0.0 <= u2 <= 1.0
    assert 0.0 <= p2 <= 1.0
