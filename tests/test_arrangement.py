"""Arrangement-control tests (plan A5). CPU-only via the mock retriever + synthetic cards.

Cannot run real ColPali locally, so these exercise the SAME core function ``run_arrangement`` with
``MockRetriever`` and tiny synthetic ID cards. The NULL prediction (MaxSim is permutation-invariant,
so shuffling patch order cannot change recovery) holds for ANY retriever, including the mock — so the
mock is a faithful test of the invariant the experiment asserts.
"""

import numpy as np

from experiments.arrangement_control import (
    permute_image_block,
    run_arrangement,
)
from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
from patchguard.retrievers.mock import MockRetriever


def test_permute_image_block_preserves_rows_and_reorders():
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((10, 4)).astype(np.float32)
    n_prefix, n_image = 1, 8  # rows 1..8 are the image block; row 0 prefix, row 9 trailing
    out, perm = permute_image_block(patches, n_prefix, n_image, rng)
    assert out.shape == patches.shape
    # prefix + trailing untouched
    assert np.array_equal(out[0], patches[0])
    assert np.array_equal(out[9], patches[9])
    # image block is a permutation of the original block (same multiset of rows)
    orig_block = patches[1:9]
    new_block = out[1:9]
    assert np.array_equal(new_block, orig_block[perm])
    assert sorted(new_block.sum(axis=1).round(5)) == sorted(orig_block.sum(axis=1).round(5))
    assert not np.array_equal(new_block, orig_block)  # actually reordered (perm != identity here)


def test_run_arrangement_is_null_on_mock():
    retriever = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    cards = generate_id_cards(6, seed=1, value_font_size=24, vary=True)
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    res = run_arrangement(retriever, cards, name_pool, n_distractors=20, seed=0)

    # both branches ran and produced valid rates
    assert res["n"] == 6
    assert 0.0 <= res["ordered_recovery"] <= 1.0
    assert 0.0 <= res["shuffled_recovery"] <= 1.0
    # THE NULL: shuffling patch order cannot change MaxSim -> identical recovery, delta exactly 0
    assert res["ordered_recovery"] == res["shuffled_recovery"]
    assert res["delta"] == 0.0
    assert res["null_confirmed"] is True
    assert res["per_card_agreement"] == 1.0
    # per-card the two branches must agree bit-for-bit
    for r in res["rows"]:
        assert r["ordered_top1"] == r["shuffled_top1"]


def test_run_arrangement_handles_zero_distractors():
    retriever = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    cards = generate_id_cards(3, seed=2, value_font_size=24, vary=True)
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    res = run_arrangement(retriever, cards, name_pool, n_distractors=0, seed=0)
    # lineup is just the true name => recovery is trivially 1.0 in both branches, still null.
    assert res["ordered_recovery"] == 1.0
    assert res["null_confirmed"] is True
