import numpy as np

from patchguard.eval.killgate import GO, MEASUREMENT, STOP, assemble_and_gate, kill_gate


def test_kill_gate_go():
    r = kill_gate(pfrr_colpali=0.65, pfrr_bipali=0.50, auc_diff_point=0.05, auc_ci_lo=0.02, auc_ci_hi=0.08)
    assert r.delta_pass and r.frontier_pass
    assert r.decision == GO
    assert abs(r.pfrr_delta_pp - 15.0) < 1e-9


def test_kill_gate_measurement_when_frontier_ci_includes_zero():
    r = kill_gate(pfrr_colpali=0.70, pfrr_bipali=0.50, auc_diff_point=0.01, auc_ci_lo=-0.01, auc_ci_hi=0.03)
    assert r.delta_pass and not r.frontier_pass
    assert r.decision == MEASUREMENT


def test_kill_gate_stop_when_delta_too_small():
    r = kill_gate(pfrr_colpali=0.55, pfrr_bipali=0.50, auc_diff_point=0.05, auc_ci_lo=0.02, auc_ci_hi=0.08)
    assert not r.delta_pass
    assert r.decision == STOP


def test_kill_gate_boundary_exactly_15pp_passes():
    r = kill_gate(pfrr_colpali=0.65, pfrr_bipali=0.50, auc_diff_point=0.01, auc_ci_lo=0.001, auc_ci_hi=0.02)
    assert r.delta_pass  # >= is inclusive


def _exact(n: int, rate: float) -> np.ndarray:
    """Deterministic recovery array with exactly floor(rate*n) ones (no sampling noise)."""
    a = np.zeros(n, dtype=float)
    a[: int(round(rate * n))] = 1.0
    return a


def _synthetic(n_docs=80, n_noise=6, colpali_rate=0.65, bipali_rate=0.45, patch_better=True, seed=0):
    rng = np.random.default_rng(seed)
    rec_colpali = _exact(n_docs, colpali_rate)
    rec_bipali = _exact(n_docs, bipali_rate)
    # utility decreases with noise; flat degrades faster than patch-scoped
    noise = np.linspace(0, 1, n_noise)
    util_patch = np.clip(1.0 - 0.10 * noise + 0.01 * rng.standard_normal((n_docs, n_noise)), 0, 1)
    util_flat = np.clip(1.0 - 0.35 * noise + 0.01 * rng.standard_normal((n_docs, n_noise)), 0, 1)
    # privacy increases with noise; patch-scoped reaches higher privacy at equal utility
    base_priv = 0.3 + 0.6 * noise
    bump = 0.15 if patch_better else -0.15
    priv_patch = np.clip(base_priv + bump + 0.01 * rng.standard_normal((n_docs, n_noise)), 0, 1)
    priv_flat = np.clip(base_priv + 0.01 * rng.standard_normal((n_docs, n_noise)), 0, 1)
    return rec_colpali, rec_bipali, util_patch, priv_patch, util_flat, priv_flat


def test_assemble_and_gate_go_on_clean_separation():
    rc, rb, up, pp, uf, pf = _synthetic(patch_better=True, seed=1)
    r = assemble_and_gate(rc, rb, up, pp, uf, pf, n_resamples=800, seed=2)
    assert r.decision == GO
    assert r.auc_ci[0] > 0  # frontier CI excludes zero on the positive side


def test_assemble_and_gate_stop_when_no_architecture_gap():
    # ColPali and BiPali recover equally -> delta ~0 -> STOP regardless of frontier
    rc, rb, up, pp, uf, pf = _synthetic(colpali_rate=0.5, bipali_rate=0.5, seed=3)
    r = assemble_and_gate(rc, rb, up, pp, uf, pf, n_resamples=800, seed=4)
    assert r.decision == STOP


def test_assemble_and_gate_shape_validation():
    rc, rb, up, pp, uf, pf = _synthetic()
    try:
        assemble_and_gate(rc, rb, up, pp[:, :3], uf, pf)  # mismatched noise dim
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")
