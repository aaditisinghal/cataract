import numpy as np

from patchguard.retrievers.base import PageEncoding, Retriever, maxsim
from patchguard.retrievers.mock import MockRetriever


def test_mock_satisfies_protocol():
    r = MockRetriever()
    assert isinstance(r, Retriever)


def test_encode_page_shapes_and_invariants():
    r = MockRetriever(grid=(4, 4), dim=8, n_prefix_tokens=1)
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    enc = r.encode_page(img)
    assert enc.patches.shape == (1 + 16, 8)
    assert enc.n_patches == 17
    assert enc.dim == 8
    assert enc.image_patches().shape == (16, 8)


def test_encode_page_deterministic():
    r = MockRetriever()
    img = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    a = r.encode_page(img).patches
    b = r.encode_page(img).patches
    assert np.allclose(a, b)


def test_page_encoding_rejects_bad_patch_count():
    try:
        PageEncoding(
            patches=np.zeros((10, 4)), grid=(4, 4), input_size=(8, 8), model_id="x"
        )  # 10 != 16
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched patch count")


def test_maxsim_matches_manual():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    # query token 0 best-matches patch 0 (dot=1); token 1 best-matches patch 2 (dot=1). sum=2.
    assert maxsim(q, p) == 2.0


def test_maxsim_dim_mismatch_raises():
    try:
        maxsim(np.zeros((2, 3)), np.zeros((4, 5)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on dim mismatch")


def test_score_path_runs():
    r = MockRetriever()
    img = (np.random.default_rng(2).random((32, 32, 3)) * 255).astype(np.uint8)
    enc = r.encode_page(img)
    s = r.score(r.encode_query("gst invoice"), enc)
    assert isinstance(s, float)
