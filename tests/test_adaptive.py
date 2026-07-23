"""CPU tests for the adaptive white-box attacker (experiments/adaptive_attack.py).

Everything runs on tiny random unit-normalised patches + a tiny RedactionProjection (dim=8) + the mock
retriever's encode_query as the query oracle. We only assert that each strategy RUNS, returns a recovery
float in [0, 1], and produces the right shapes — not any particular science number (that needs ColPali/GPU).
"""

import numpy as np
import torch

from patchguard.defense.redact import RedactionProjection
from patchguard.retrievers.mock import MockRetriever

from experiments.adaptive_attack import (
    Card,
    _apply_P,
    _lineup,
    baseline_dictionary,
    strat_adaptive_query,
    strat_correlation,
    strat_distillation,
    strat_inverse,
)

DIM = 8
NAMES = ["ALICE ADAMS", "BOB BROWN", "CARY CLARK", "DINA DOE"]
TOPICS = ["DISTRICT 11 SALEM OFFICE", "DISTRICT 22 DOVER OFFICE",
          "DISTRICT 33 MILTON OFFICE", "DISTRICT 44 NEWPORT OFFICE"]


def _mk_cards(n, seed, shared_labels=True):
    """Tiny synthetic cards: random unit-norm patches, closed 4-name label space."""
    rng = np.random.default_rng(seed)
    cards = []
    for i in range(n):
        idx = i % len(NAMES) if shared_labels else int(rng.integers(0, len(NAMES)))
        p = rng.standard_normal((6, DIM)).astype(np.float32)
        p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
        cards.append(Card(patches=p, name=NAMES[idx], name_idx=idx, topic=TOPICS[idx]))
    return cards


def _q_fn():
    mock = MockRetriever(dim=DIM)
    return mock.encode_query


def _in_unit(x):
    return isinstance(x, float) and 0.0 - 1e-9 <= x <= 1.0 + 1e-9


def test_helpers_shapes():
    P = RedactionProjection(dim=DIM)
    c = _mk_cards(1, 0)[0]
    out = _apply_P(P, c.patches, "cpu")
    assert out.shape == (6, DIM)
    assert torch.allclose(out.norm(dim=-1), torch.ones(6), atol=1e-5)
    lu = _lineup(NAMES[0], NAMES, 2, np.random.default_rng(0))
    assert lu[0] == NAMES[0] and len(lu) == 3


def test_baselines_run():
    P = RedactionProjection(dim=DIM)
    cards = _mk_cards(6, 1)
    qf = _q_fn()
    defended = baseline_dictionary(cards, P, qf, NAMES, 2, "cpu", np.random.default_rng(0))
    undef = baseline_dictionary(cards, None, qf, NAMES, 2, "cpu", np.random.default_rng(0))
    assert _in_unit(defended) and _in_unit(undef)


def test_strat_distillation_runs():
    P = RedactionProjection(dim=DIM)
    train, test = _mk_cards(8, 2), _mk_cards(4, 3)
    rec, m = strat_distillation(train, test, P, n_classes=len(NAMES), device="cpu", epochs=5, seed=0)
    assert _in_unit(rec)
    assert _in_unit(m["mlp_top1"]) and _in_unit(m["linear_top1"])


def test_strat_inverse_runs():
    P = RedactionProjection(dim=DIM)
    pairs, test = _mk_cards(6, 4), _mk_cards(4, 5)
    rec, m = strat_inverse(pairs, test, P, _q_fn(), NAMES, 2, "cpu", epochs=5, seed=0,
                           rng=np.random.default_rng(0))
    assert _in_unit(rec)
    assert m["n_pairs"] == 6 * 6  # 6 cards * 6 patches each


def test_strat_adaptive_query_runs():
    P = RedactionProjection(dim=DIM)
    test = _mk_cards(3, 6)
    rec, m = strat_adaptive_query(test, P, _q_fn(), NAMES, 2, "cpu", epochs=3, lr=0.05,
                                  anchor=0.1, seed=0, rng=np.random.default_rng(0))
    assert _in_unit(rec)
    assert m["epochs"] == 3


def test_strat_correlation_runs():
    P = RedactionProjection(dim=DIM)
    train, test = _mk_cards(8, 7), _mk_cards(4, 8)
    rec, m = strat_correlation(train, test, P, _q_fn(), TOPICS, NAMES, 2, n_classes=len(NAMES),
                               device="cpu", epochs=5, seed=0, rng=np.random.default_rng(0))
    assert _in_unit(rec)
    assert _in_unit(m["topic_recovery"])
