"""Pooled BiPali control: single-vector page, and 'no locality' via a 1x1 grid."""

import numpy as np

from patchguard.data.align import boxes_to_patch_mask
from patchguard.retrievers.base import Retriever
from patchguard.retrievers.bipali import PooledRetriever
from patchguard.retrievers.mock import MockRetriever


def _img(seed=0):
    return (np.random.default_rng(seed).random((32, 32, 3)) * 255).astype(np.uint8)


def test_pooled_satisfies_protocol():
    assert isinstance(PooledRetriever(MockRetriever()), Retriever)


def test_pooled_page_is_single_vector():
    r = PooledRetriever(MockRetriever(grid=(4, 4), dim=8))
    enc = r.encode_page(_img())
    assert enc.patches.shape == (1, 8)
    assert enc.grid == (1, 1)
    assert enc.n_patches == 1
    # unit-normalized
    assert abs(float(np.linalg.norm(enc.patches[0])) - 1.0) < 1e-5


def test_pooled_has_no_locality():
    # On a 1x1 grid, ANY field box selects the single page vector — the defining property.
    r = PooledRetriever(MockRetriever(grid=(4, 4), dim=8))
    enc = r.encode_page(_img())
    mask_corner = boxes_to_patch_mask([(0, 0, 4, 4)], (32, 32), enc.grid, enc.input_size)
    mask_elsewhere = boxes_to_patch_mask([(20, 20, 28, 28)], (32, 32), enc.grid, enc.input_size)
    assert mask_corner.tolist() == [True]
    assert mask_elsewhere.tolist() == [True]  # different box, same single vector


def test_pooled_score_runs():
    r = PooledRetriever(MockRetriever())
    enc = r.encode_page(_img())
    s = r.score(r.encode_query("invoice total"), enc)
    assert isinstance(s, float)
