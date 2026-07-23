"""CPU tests for the certified impossibility bound (experiments/certified_bound.py).

These pin the geometry the certified claim rests on, with no ColPali/GPU:
  * the index-time map is an ORTHOGONAL PROJECTION: idempotent and of rank exactly d-k;
  * the OPTIMAL linear reconstruction (Moore-Penrose pseudo-inverse of the projection) cannot reduce the
    in-span(D) error below the removed energy — it leaves ~100% of it (span recovery == chance), while the
    complement span(D)^⊥ is recovered exactly.
"""

import numpy as np
import torch

from experiments.certified_bound import (
    _projection,
    _random_orthonormal,
    _span_metrics,
    optimal_linear_inverse_eval,
)


def _unit_patches(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float32)
    return X / X.norm(dim=-1, keepdim=True)


def test_projection_idempotent_and_rank():
    d, k = 16, 4
    D = _random_orthonormal(d, k, seed=1)
    P = _projection(D)
    # idempotent: P @ P == P (defining property of an orthogonal projection)
    assert torch.allclose(P @ P, P, atol=1e-5)
    # symmetric
    assert torch.allclose(P, P.T, atol=1e-5)
    # rank exactly d-k
    assert int(torch.linalg.matrix_rank(P)) == d - k


def test_random_orthonormal_columns():
    D = _random_orthonormal(20, 5, seed=3)
    assert D.shape == (20, 5)
    assert torch.allclose(D.T @ D, torch.eye(5), atol=1e-4)
    assert _random_orthonormal(20, 0, seed=3).shape == (20, 0)


def test_optimal_inverse_cannot_beat_removed_energy():
    # The optimal linear inverse leaves ~100% of the span(D) energy as error (== full loss).
    d, k, n = 24, 6, 400
    X = _unit_patches(n, d, seed=7)
    D = _random_orthonormal(d, k, seed=2)
    m = optimal_linear_inverse_eval(X, D)
    # in-span error / removed energy ~ 1.0 : the adversary recovers NONE of span(D)
    assert abs(m["span_error_fraction"] - 1.0) < 1e-3
    assert abs(m["span_recovery"] - 0.0) < 1e-3           # == chance
    # projection diagnostics carried through
    assert m["projection_rank"] == d - k == m["expected_rank"]
    assert m["projection_idempotent_err"] < 1e-4
    assert m["pinv_equals_projection_err"] < 1e-4         # pinv(P) == P for a projection
    # the complement is recovered exactly (utility is untouched)
    assert m["out_recovered"] > 0.999


def test_no_linear_map_recovers_span_via_ols():
    # Even the data-fit least-squares map Y->X leaves ~all span(D) energy as error (n >> d, no overfit).
    d, k, n = 16, 4, 3000
    X = _unit_patches(n, d, seed=11)
    D = _random_orthonormal(d, k, seed=5)
    P = _projection(D)
    Y = X @ P
    A = torch.linalg.lstsq(Y, X).solution          # best linear map X ~= Y @ A
    Xhat = Y @ A
    m = _span_metrics(X, Xhat, D)
    assert m["span_error_fraction"] >= 0.95         # cannot meaningfully beat the removed-energy floor


def test_k_zero_control():
    d, n = 12, 100
    X = _unit_patches(n, d, seed=9)
    D = _random_orthonormal(d, 0, seed=1)
    m = optimal_linear_inverse_eval(X, D)
    # nothing removed -> trivial perfect reconstruction, span recovery defined as 1.0 (no protection)
    assert m["removed_energy_fraction"] == 0.0
    assert m["span_recovery"] == 1.0
    assert m["projection_rank"] == d


def test_removed_energy_grows_with_k():
    d, n = 32, 500
    X = _unit_patches(n, d, seed=4)
    fracs = []
    for k in (2, 8, 16):
        D = _random_orthonormal(d, k, seed=6)
        fracs.append(optimal_linear_inverse_eval(X, D)["removed_energy_fraction"])
    assert fracs[0] < fracs[1] < fracs[2]           # more removed directions -> more removed energy
