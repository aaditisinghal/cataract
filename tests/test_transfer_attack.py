"""CPU tests for the cross-encoder transfer attack (experiments/transfer_attack.py).

Two independent MockRetrievers with the SAME dim but DIFFERENT seeds stand in for ColPali and ColQwen2:
they produce distinct (unaligned) embedding spaces, so both the matched (same encoder) and the transfer
(cross encoder + learned linear alignment) code paths are genuinely exercised. We assert only that each
path RUNS and returns a recovery in [0, 1] and that ``matched >= transfer`` is COMPUTED as a boolean --
never that transfer actually degrades, which is an empirical question for the real GPU run.
"""

import numpy as np

from patchguard.retrievers.mock import MockRetriever

from experiments.transfer_attack import (
    _beats_chance,
    _recover,
    align_query,
    fit_alignment,
    run_transfer_pair,
)

DIM = 8
NAMES = ["ALICE ADAMS", "BOB BROWN", "CARY CLARK", "DINA DOE", "EVE EAST", "FRED FOX"]
ANCHORS = [f"DISTRICT {10 + i} SALEM OFFICE" for i in range(12)] + \
          [f"{1970 + i}" for i in range(12)]


def _mk_cards(n, seed):
    """Tiny (H, W, 3) uint8 images tagged with a ground-truth name from the closed pool."""
    rng = np.random.default_rng(seed)
    cards = []
    for i in range(n):
        img = (rng.random((24, 24, 3)) * 255).astype(np.uint8)
        cards.append((img, NAMES[i % len(NAMES)]))
    return cards


def _in_unit(x):
    return isinstance(x, float) and -1e-9 <= x <= 1.0 + 1e-9


def test_fit_and_align_shapes():
    src = MockRetriever(dim=DIM, seed=1)
    dst = MockRetriever(dim=DIM, seed=2)
    W = fit_alignment(ANCHORS, src.encode_query, dst.encode_query, ridge=1e-2)
    assert W.shape == (DIM, DIM)  # (d_dst, d_src)
    q = src.encode_query("ALICE ADAMS")
    mapped = align_query(q, W)
    assert mapped.shape == q.shape
    # rows are unit-normalised after alignment
    norms = np.linalg.norm(mapped, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_recover_in_unit():
    ret = MockRetriever(dim=DIM, seed=3)
    cards = _mk_cards(5, 10)
    idx_cards = [(np.asarray(ret.encode_page(im).patches, np.float32), nm) for im, nm in cards]
    rec = _recover(idx_cards, ret.encode_query, NAMES, n_distractors=3, seed=0)
    assert _in_unit(rec)


def test_transfer_pair_matched_and_transfer_run():
    index_ret = MockRetriever(dim=DIM, seed=1)
    query_ret = MockRetriever(dim=DIM, seed=2)
    cards = _mk_cards(6, 20)
    res = run_transfer_pair(index_ret, query_ret, cards, NAMES, ANCHORS,
                            n_distractors=3, lineup_seed=0, ridge=1e-2)

    assert _in_unit(res["matched_recovery"])
    assert _in_unit(res["transfer_recovery"])
    # matched >= transfer must be COMPUTED (a bool), NOT asserted as ground truth.
    assert isinstance(res["matched_ge_transfer"], bool)
    assert res["matched_ge_transfer"] == (res["matched_recovery"] >= res["transfer_recovery"])
    assert np.isclose(res["degradation"], res["matched_recovery"] - res["transfer_recovery"])
    assert res["dim_match"] is True and res["d_index"] == DIM and res["d_query"] == DIM
    # same-dim mocks -> the raw unaligned floor is also computed and valid
    assert _in_unit(res["transfer_raw_unaligned_recovery"])
    assert isinstance(res["transfer_beats_chance"], bool)
    assert 0.0 < res["chance"] <= 1.0


def test_transfer_pair_reverse_direction_runs():
    index_ret = MockRetriever(dim=DIM, seed=2)
    query_ret = MockRetriever(dim=DIM, seed=1)
    cards = _mk_cards(4, 30)
    res = run_transfer_pair(index_ret, query_ret, cards, NAMES, ANCHORS,
                            n_distractors=2, lineup_seed=0, ridge=1e-2)
    assert _in_unit(res["matched_recovery"]) and _in_unit(res["transfer_recovery"])
    assert isinstance(res["matched_ge_transfer"], bool)


def test_beats_chance_logic():
    assert _beats_chance(0.9, 0.01) is True
    assert _beats_chance(0.02, 0.01) is False  # above chance but tiny in absolute terms
    assert _beats_chance(0.001, 0.005) is False
