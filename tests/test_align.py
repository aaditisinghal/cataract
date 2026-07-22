"""Geometry tests for the crux module. Hand-computed expectations; no model needed."""

import numpy as np

from patchguard.data.align import boxes_to_patch_mask, patch_index, to_resized_coords


def _selected(mask, grid, n_prefix=0):
    """Return set of (row, col) selected image patches from a flat mask."""
    gh, gw = grid
    img = mask[n_prefix:].reshape(gh, gw)
    return {(int(i), int(j)) for i in range(gh) for j in range(gw) if img[i, j]}


def test_single_patch_no_scaling():
    # 4x4 grid over an 8x8 input; each patch is 2x2 px. orig==input => no scaling.
    # Box (0,0,2,2) covers exactly patch (0,0).
    mask = boxes_to_patch_mask(
        boxes=[(0, 0, 2, 2)], orig_size=(8, 8), grid=(4, 4), input_size=(8, 8)
    )
    assert _selected(mask, (4, 4)) == {(0, 0)}


def test_2x2_block_of_patches():
    # Box (0,0,4,4) covers the top-left 2x2 block of patches.
    mask = boxes_to_patch_mask(
        boxes=[(0, 0, 4, 4)], orig_size=(8, 8), grid=(4, 4), input_size=(8, 8)
    )
    assert _selected(mask, (4, 4)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_row_major_indexing_matches_helper():
    # A box on patch (row=1, col=2): x in [4,6], y in [2,4].
    mask = boxes_to_patch_mask(
        boxes=[(4, 2, 6, 4)], orig_size=(8, 8), grid=(4, 4), input_size=(8, 8)
    )
    assert _selected(mask, (4, 4)) == {(1, 2)}
    idx = patch_index(1, 2, (4, 4))
    assert idx == 1 * 4 + 2
    assert mask[idx]


def test_prefix_tokens_are_never_selected_and_shift_indices():
    mask = boxes_to_patch_mask(
        boxes=[(0, 0, 2, 2)],
        orig_size=(8, 8),
        grid=(4, 4),
        input_size=(8, 8),
        n_prefix_tokens=3,
    )
    assert mask.shape[0] == 3 + 16
    assert not mask[:3].any()  # prefix tokens never spatial
    assert mask[patch_index(0, 0, (4, 4), n_prefix_tokens=3)]


def test_coverage_threshold_excludes_slivers():
    # Box (0,0,3,2): column 0 fully covered (2px), column 1 covered 1px of 2 => coverage 0.5.
    grid, inp, orig = (4, 4), (8, 8), (8, 8)
    any_overlap = _selected(
        boxes_to_patch_mask([(0, 0, 3, 2)], orig, grid, inp, coverage_threshold=0.0), grid
    )
    assert (0, 1) in any_overlap  # sliver counts at threshold 0
    strict = _selected(
        boxes_to_patch_mask([(0, 0, 3, 2)], orig, grid, inp, coverage_threshold=0.6), grid
    )
    assert (0, 1) not in strict  # 0.5 coverage < 0.6 threshold => dropped
    assert (0, 0) in strict  # full coverage still counts


def test_squash_scaling():
    # Original 16x8 squashed into 8x8 input => sx=0.5, sy=1.0. Box (0,0,4,2)->(0,0,2,2)=patch(0,0).
    coords = to_resized_coords((0, 0, 4, 2), orig_size=(16, 8), input_size=(8, 8))
    assert coords == (0.0, 0.0, 2.0, 2.0)
    mask = boxes_to_patch_mask([(0, 0, 4, 2)], (16, 8), (4, 4), (8, 8))
    assert _selected(mask, (4, 4)) == {(0, 0)}


def test_letterbox_padding_centers_content():
    # Original 8x4 into 8x8 input, letterbox: s=min(1.0,2.0)=1.0, new=(8,4), pad_y=2.
    # Box (0,0,8,4) -> (0,2,8,6): rows 1..2 (y in [2,6] over 2px rows) fully covered, cols all.
    coords = to_resized_coords((0, 0, 8, 4), orig_size=(8, 4), input_size=(8, 8), resize_policy="letterbox")
    assert coords == (0.0, 2.0, 8.0, 6.0)
    mask = boxes_to_patch_mask(
        [(0, 0, 8, 4)], (8, 4), (4, 4), (8, 8), resize_policy="letterbox"
    )
    sel = _selected(mask, (4, 4))
    assert sel == {(1, j) for j in range(4)} | {(2, j) for j in range(4)}


def test_degenerate_box_selects_nothing():
    mask = boxes_to_patch_mask([(5, 5, 5, 5)], (8, 8), (4, 4), (8, 8))
    assert not mask.any()
