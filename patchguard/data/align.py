"""Map field bounding boxes (original-image pixels) -> patch indices (MASTER_PLAN S3).

THE crux module and the #1 silent risk (RESEARCH_PROTOCOL S11): an off-by-one here corrupts every
defense number invisibly. So the geometry is explicit and every convention is stated.

Pipeline for one box:
  1. Transform box from original pixel coords -> the resized (model input) coord space, using the
     SAME resize policy the model's image processor used (squash = independent x/y scale;
     letterbox = uniform scale + centered padding). Read the real policy from the processor config;
     do not assume.
  2. Each patch (row i, col j) covers a rectangle in resized space:
        x in [j*pw, (j+1)*pw],  y in [i*ph, (i+1)*ph],   pw = iw/gw, ph = ih/gh.
  3. Coverage of a patch = (box∩patch area) / (patch area). Patch is selected when
        coverage > coverage_threshold.   threshold=0.0 => any positive overlap counts (conservative:
     protects more, the default for the defense). Report sensitivity at 0.0/0.25/0.5 (appendix).
  4. Flatten ROW-MAJOR and prepend ``n_prefix_tokens`` False entries so the mask aligns to the
     model's patch sequence, where patch (i,j) sits at index ``n_prefix + i*gw + j``.

Conventions (must match PageEncoding): grid=(rows,cols)=(gh,gw); input_size=(width,height)=(iw,ih);
boxes are (x0, y0, x1, y1) with x0<=x1, y0<=y1 in original-image pixels.
"""

from __future__ import annotations

import numpy as np

# (x0, y0, x1, y1) in original-image pixel coordinates.
Box = tuple[float, float, float, float]

_VALID_POLICIES = ("squash", "letterbox")


def to_resized_coords(
    box: Box,
    orig_size: tuple[int, int],
    input_size: tuple[int, int],
    resize_policy: str = "squash",
) -> Box:
    """Map a box from original pixel space into the model's resized (input) pixel space."""
    ow, oh = orig_size
    iw, ih = input_size
    x0, y0, x1, y1 = box
    if resize_policy == "squash":
        sx, sy = iw / ow, ih / oh
        return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
    if resize_policy == "letterbox":
        s = min(iw / ow, ih / oh)
        new_w, new_h = ow * s, oh * s
        pad_x, pad_y = (iw - new_w) / 2.0, (ih - new_h) / 2.0
        return (x0 * s + pad_x, y0 * s + pad_y, x1 * s + pad_x, y1 * s + pad_y)
    raise ValueError(f"resize_policy must be one of {_VALID_POLICIES}; got {resize_policy!r}")


def patch_index(row: int, col: int, grid: tuple[int, int], n_prefix_tokens: int = 0) -> int:
    """Sequence index of image patch (row, col). Row-major, prefix tokens counted."""
    gh, gw = grid
    if not (0 <= row < gh and 0 <= col < gw):
        raise IndexError(f"patch ({row},{col}) out of grid {grid}")
    return n_prefix_tokens + row * gw + col


def boxes_to_patch_mask(
    boxes: list[Box],
    orig_size: tuple[int, int],
    grid: tuple[int, int],
    input_size: tuple[int, int],
    resize_policy: str = "squash",
    coverage_threshold: float = 0.0,
    n_prefix_tokens: int = 0,
) -> np.ndarray:
    """Boolean mask over the patch sequence: True where a patch is covered by any box.

    Returns shape ``(n_prefix_tokens + gh*gw,)``. Prefix-token positions are always False (they carry
    no spatial location). ``coverage_threshold`` in [0, 1): 0.0 selects on any positive overlap.
    """
    if not (0.0 <= coverage_threshold < 1.0):
        raise ValueError("coverage_threshold must be in [0, 1)")
    gh, gw = grid
    iw, ih = input_size
    pw, ph = iw / gw, ih / gh
    patch_area = pw * ph

    # Patch edge coordinates in resized space.
    col_x0 = np.arange(gw) * pw  # (gw,)
    col_x1 = col_x0 + pw
    row_y0 = np.arange(gh) * ph  # (gh,)
    row_y1 = row_y0 + ph

    mask_grid = np.zeros((gh, gw), dtype=bool)
    for box in boxes:
        rx0, ry0, rx1, ry1 = to_resized_coords(box, orig_size, input_size, resize_policy)
        if rx1 <= rx0 or ry1 <= ry0:
            continue  # degenerate box
        # 1-D overlap of the box with each column / row.
        ox = np.clip(np.minimum(rx1, col_x1) - np.maximum(rx0, col_x0), 0.0, None)  # (gw,)
        oy = np.clip(np.minimum(ry1, row_y1) - np.maximum(ry0, row_y0), 0.0, None)  # (gh,)
        overlap = np.outer(oy, ox)  # (gh, gw) intersection area
        coverage = overlap / patch_area
        mask_grid |= coverage > coverage_threshold

    flat = mask_grid.reshape(-1)  # row-major: index = i*gw + j
    if n_prefix_tokens:
        flat = np.concatenate([np.zeros(n_prefix_tokens, dtype=bool), flat])
    return flat
