"""Defense-primitive tests (Claim 3 core): perturbation locality + oracle localization."""

import numpy as np

from patchguard.defense.localize import OracleLocalizer
from patchguard.defense.perturb import flat_gaussian, patch_scoped_gaussian
from patchguard.retrievers.mock import MockRetriever


def _enc(seed=0, grid=(4, 4), dim=8, prefix=1):
    r = MockRetriever(grid=grid, dim=dim, n_prefix_tokens=prefix)
    img = (np.random.default_rng(seed).random((32, 32, 3)) * 255).astype(np.uint8)
    return r.encode_page(img)


def test_flat_gaussian_changes_all_image_patches():
    enc = _enc()
    out = flat_gaussian(enc, sigma=0.3, seed=1)
    img_before = enc.image_patches()
    img_after = out.image_patches()
    # every image patch moved
    assert not np.allclose(img_before, img_after)
    changed = ~np.all(np.isclose(img_before, img_after), axis=1)
    assert changed.all()


def test_flat_gaussian_leaves_prefix_untouched():
    enc = _enc(prefix=1)
    out = flat_gaussian(enc, sigma=0.5, seed=2)
    assert np.allclose(enc.patches[:1], out.patches[:1])  # prefix token unchanged


def test_patch_scoped_only_touches_masked_patches():
    enc = _enc()
    # mask two specific image patches (indices in the full sequence, prefix=1)
    mask = np.zeros(enc.n_patches, dtype=bool)
    mask[3] = True
    mask[7] = True
    out = patch_scoped_gaussian(enc, mask, sigma=0.4, seed=3)
    diff = ~np.all(np.isclose(enc.patches, out.patches), axis=1)
    assert diff.tolist().count(True) == 2
    assert diff[3] and diff[7]


def test_patch_scoped_perturbs_fewer_than_flat():
    enc = _enc()
    mask = np.zeros(enc.n_patches, dtype=bool)
    mask[3] = True
    scoped = patch_scoped_gaussian(enc, mask, sigma=0.4, seed=4)
    flat = flat_gaussian(enc, sigma=0.4, seed=4)
    n_scoped = int((~np.all(np.isclose(enc.patches, scoped.patches), axis=1)).sum())
    n_flat = int((~np.all(np.isclose(enc.patches, flat.patches), axis=1)).sum())
    assert n_scoped < n_flat  # the whole point: touch 1, not 16


def test_perturbed_patches_stay_unit_norm():
    enc = _enc()
    out = flat_gaussian(enc, sigma=0.9, seed=5, renormalize=True)
    norms = np.linalg.norm(out.image_patches(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mask_length_mismatch_raises():
    enc = _enc()
    try:
        patch_scoped_gaussian(enc, np.zeros(3, dtype=bool), sigma=0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mask length mismatch")


def test_oracle_localizer_flags_field_patches():
    # 4x4 grid over 8x8 input, prefix=1. A field box on the top-left 2x2 block.
    r = MockRetriever(grid=(4, 4), input_size=(8, 8), dim=8, n_prefix_tokens=1)
    img = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
    enc = r.encode_page(img)
    loc = OracleLocalizer(coverage_threshold=0.0)
    mask = loc.locate(enc, field_boxes=[(0, 0, 4, 4)], orig_size=(8, 8))
    assert mask.shape[0] == enc.n_patches
    assert not mask[0]  # prefix token never flagged
    # image patches (0,0),(0,1),(1,0),(1,1) -> sequence idx 1 + {0,1,4,5}
    flagged = set(np.nonzero(mask)[0].tolist())
    assert flagged == {1 + 0, 1 + 1, 1 + 4, 1 + 5}


def test_oracle_then_scoped_perturb_pipeline():
    # end-to-end: locate PII patches, perturb only those.
    r = MockRetriever(grid=(4, 4), input_size=(8, 8), dim=8, n_prefix_tokens=1)
    img = (np.random.default_rng(1).random((8, 8, 3)) * 255).astype(np.uint8)
    enc = r.encode_page(img)
    mask = OracleLocalizer().locate(enc, [(0, 0, 2, 2)], (8, 8))  # one patch
    out = patch_scoped_gaussian(enc, mask, sigma=0.5, seed=0)
    changed = int((~np.all(np.isclose(enc.patches, out.patches), axis=1)).sum())
    assert changed == int(mask.sum()) == 1
