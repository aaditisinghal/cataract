"""CPU tests for experiments/margin_analysis.py — mock retriever + tiny patches, no GPU/ColPali.

Checks: (1) the deletion budget is monotone in the target count (erasure mechanism reused correctly);
(2) the margin sweep returns a well-formed, finite per-fraction record with a monotone-non-decreasing
removed-fraction list; (3) the honest verdict read-out runs.
"""

import numpy as np

from experiments.margin_analysis import _deletion_mask, _dilate, _verdict, margin_sweep
from patchguard.data.fields import AnnotatedField
from patchguard.retrievers.mock import MockRetriever

# 4x4 grid over a 32x32 page (MockRetriever defaults) -> each patch is 8x8 px.
_GRID = (4, 4)
_NAMES = ["JAMES SMITH", "MARIA JONES", "ROBERT BROWN", "LINDA DAVIS", "DAVID MILLER", "SARAH WILSON"]


def _tiny_card(name, seed):
    rng = np.random.default_rng(seed)
    img = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    # Name box covers patches (0,0) and (0,1): x in [0,16], y in [0,8] on the 8x8-px patch grid.
    field = AnnotatedField(field_type="name", text=name, box=(0.0, 0.0, 16.0, 8.0))
    return img, [field]


def test_dilate_grows_monotonically():
    m = np.zeros(_GRID[0] * _GRID[1], dtype=bool)
    m[0] = True  # patch (0,0)
    counts = [_dilate(m, _GRID, r).sum() for r in range(4)]
    assert counts[0] == 1
    assert counts == sorted(counts)  # non-decreasing


def test_deletion_mask_monotone_in_budget():
    field = np.zeros(_GRID[0] * _GRID[1], dtype=bool)
    field[0] = field[1] = True  # a 2-patch field
    prev = -1
    for target in [0, 1, 2, 4, 8, 16]:
        cnt = int(_deletion_mask(field, _GRID, target).sum())
        assert cnt >= prev  # budget monotone
        prev = cnt
    assert _deletion_mask(field, _GRID, 0).sum() == 0  # fraction=0 deletes nothing
    # Below the field size we still delete the whole (indivisible) field.
    assert _deletion_mask(field, _GRID, 1).sum() == 2
    assert _deletion_mask(field, _GRID, 16).sum() == 16  # budget can consume the grid


def test_margin_sweep_shapes_and_monotone_removal():
    retriever = MockRetriever()  # grid=(4,4), input_size=(32,32), n_prefix_tokens=1
    cards = [_tiny_card(_NAMES[i % len(_NAMES)], seed=i) for i in range(4)]
    fractions = [0.0, 0.1, 0.2, 0.33]
    out = margin_sweep(retriever, cards, _NAMES, fractions, distractors=4, seed=0)

    assert out["n"] == 4
    pf = out["per_fraction"]
    assert len(pf) == len(fractions)

    removed_seq = []
    for rec, frac in zip(pf, fractions):
        assert rec["fraction"] == frac
        assert np.isfinite(rec["margin_mean"])          # margin computed
        assert len(rec["margin_ci"]) == 2
        assert all(np.isfinite(c) for c in rec["margin_ci"])
        assert 0.0 <= rec["top1"] <= 1.0
        removed_seq.append(rec["removed_frac_mean"])

    # The removed-fraction list is a monotone (non-decreasing) sweep — the core "monotone-ish" property.
    assert removed_seq == sorted(removed_seq)
    assert removed_seq[0] == 0.0                          # fraction 0 = full-page baseline
    assert removed_seq[-1] > 0.0                          # deletion actually happened


def test_verdict_runs_on_sweep():
    retriever = MockRetriever()
    cards = [_tiny_card(_NAMES[i % len(_NAMES)], seed=i) for i in range(3)]
    out = margin_sweep(retriever, cards, _NAMES, [0.0, 0.2], distractors=3, seed=1)
    v = _verdict(out["per_fraction"])
    assert v["label"] in {
        "holographic_margin_persists",
        "narrows_top1_saturated_margin_decays",
        "leak_erodes",
        "no_data",
    }
    assert np.isfinite(v["margin_retention"])
