"""CPU tests for the field baselines (patchguard/defense/baselines.py) and the frontier sweep helpers.

Everything runs on the MOCK retriever + tiny tensors: no colpali/torch-model download. The two claims
these tests protect: (1) every baseline returns unit-normalized patches of the same shape whatever the
strength; (2) the reusable baseline_frontier sweep produces well-formed (utility, privacy) points.
"""

import numpy as np
import pytest

from patchguard.defense.baselines import (
    BASELINE_NAMES,
    apply_baseline,
    entroguard,
    get_baseline,
    koga,
    press,
)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _rand_patches(seed=0, npch=24, d=16):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal((npch, d)).astype(np.float32))


@pytest.mark.parametrize("name", BASELINE_NAMES)
@pytest.mark.parametrize("strength", [0.0, 0.05, 0.2, 0.5])
def test_baseline_shape_and_normalized(name, strength):
    x = _rand_patches()
    rng = np.random.default_rng(1)
    y = apply_baseline(name, x, strength, rng=rng)
    assert y.shape == x.shape
    assert y.dtype == np.float32 or y.dtype == np.float64
    norms = np.linalg.norm(y, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_press_removes_supplied_contrast_subspace():
    # If PRESS is told the PII subspace, patches should lose their component along it.
    d = 16
    x = _rand_patches(seed=2, d=d)
    contrast = np.zeros((3, d), dtype=np.float32)
    contrast[0, 0] = 1.0  # PII direction = axis 0
    y = press(x, strength=1.0 / d, contrast=contrast)  # k = round(1.0/d * d) = 1 direction removed
    # residual projection onto axis 0 should be ~0 before renormalization dominates; check pre-norm math
    # via the public transform: correlation of y with axis-0 should shrink vs x.
    assert abs(float(np.mean(y[:, 0]))) <= abs(float(np.mean(x[:, 0]))) + 1e-6
    # and the removed-direction energy is essentially gone (small relative to total)
    proj_energy = float(np.mean((y @ contrast[0]) ** 2))
    assert proj_energy < 1e-2


def test_entroguard_identity_at_zero_strength():
    x = _rand_patches(seed=3)
    y = entroguard(x, 0.0, rng=np.random.default_rng(0))
    assert np.allclose(y, x, atol=1e-5)


def test_koga_moves_patches_more_with_more_strength():
    x = _rand_patches(seed=4)
    y_lo = koga(x, 0.05, rng=np.random.default_rng(7))
    y_hi = koga(x, 0.5, rng=np.random.default_rng(7))
    assert float(np.linalg.norm(y_hi - x)) > float(np.linalg.norm(y_lo - x))


def test_get_baseline_unknown_raises():
    with pytest.raises(ValueError):
        get_baseline("nope")


def test_torch_input_roundtrips_to_torch():
    torch = pytest.importorskip("torch")
    x = torch.nn.functional.normalize(torch.randn(20, 12), dim=-1)
    y = apply_baseline("entroguard", x, 0.2, rng=np.random.default_rng(0))
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), torch.ones(20), atol=1e-4)


def test_baseline_frontier_sweep_runs_on_mock():
    from experiments.baseline_frontier import baseline_frontier, gen_cards, make_qcache
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

    pts = baseline_frontier("entroguard", test, q, pool, [0.0, 0.3], distractors=5, rng=rng)
    assert len(pts) == 2
    for p in pts:
        assert set(p) == {"strength", "utility", "privacy"}
        assert 0.0 <= p["utility"] <= 1.0
        assert 0.0 <= p["privacy"] <= 1.0


def test_frontier_util_at_priv_interpolates():
    from experiments.baseline_frontier import util_at_priv

    pts = [{"strength": 0.0, "utility": 1.0, "privacy": 0.0},
           {"strength": 0.5, "utility": 0.5, "privacy": 1.0}]
    assert abs(util_at_priv(pts, 0.5) - 0.75) < 1e-6
