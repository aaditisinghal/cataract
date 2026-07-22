"""Visual validator for patch<->field alignment (MASTER_PLAN S3 gate).

The two hours of eyeballing that save a retracted result: render page + field-box overlay + the
patch mask boxes_to_patch_mask() produces, side by side. If the highlighted patches don't sit on
the fields, STOP and fix align.py before anything downstream uses it.

Usage:
    # self-test on a synthetic page (no dataset needed — proves the pipeline before data lands):
    python -m experiments.validate_alignment --selftest --out results/align_selftest.png

    # on real samples, call render_alignment(image, boxes, grid, input_size, ...) from a loader
    # that yields 50 docs/dataset and save each overlay.

Requires the [viz] extras (matplotlib, pillow). Kept out of the CI logic core on purpose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from patchguard.data.align import Box, boxes_to_patch_mask


def render_alignment(
    image: np.ndarray,
    boxes: list[Box],
    grid: tuple[int, int],
    input_size: tuple[int, int],
    resize_policy: str = "squash",
    coverage_threshold: float = 0.0,
    n_prefix_tokens: int = 0,
    out_path: str | Path = "results/alignment.png",
    title: str = "alignment",
) -> Path:
    """Save a 2-panel figure: (left) page + field boxes, (right) page + selected patch grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    gh, gw = grid
    iw, ih = input_size
    h, w = image.shape[:2]
    pw, ph = w / gw, h / gh  # patch size in ORIGINAL pixels, for drawing on the shown image

    mask = boxes_to_patch_mask(
        boxes, (w, h), grid, input_size, resize_policy, coverage_threshold, n_prefix_tokens
    )
    img_mask = mask[n_prefix_tokens:].reshape(gh, gw)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (ax0, ax1):
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])

    ax0.set_title(f"{title}: field boxes")
    for (x0, y0, x1, y1) in boxes:
        ax0.add_patch(
            mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=2)
        )

    ax1.set_title(f"selected patches ({int(img_mask.sum())}/{gh * gw})")
    for i in range(gh):
        for j in range(gw):
            if img_mask[i, j]:
                ax1.add_patch(
                    mpatches.Rectangle(
                        (j * pw, i * ph), pw, ph, fill=True, alpha=0.35, color="lime"
                    )
                )
    # faint grid
    for j in range(gw + 1):
        ax1.axvline(j * pw, color="white", lw=0.3, alpha=0.4)
    for i in range(gh + 1):
        ax1.axhline(i * ph, color="white", lw=0.3, alpha=0.4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _selftest(out_path: str) -> None:
    """Synthetic checkerboard page with two field boxes — eyeball that green sits on red."""
    rng = np.random.default_rng(0)
    img = (rng.random((224, 224, 3)) * 60 + 190).astype(np.uint8)  # light noisy page
    # two "fields": a name row and an account-number box
    boxes: list[Box] = [(20, 30, 180, 55), (60, 150, 140, 175)]
    for x0, y0, x1, y1 in boxes:
        img[int(y0) : int(y1), int(x0) : int(x1)] = 40  # dark ink where the field is
    path = render_alignment(
        img, boxes, grid=(16, 16), input_size=(224, 224), out_path=out_path, title="selftest"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results/align_selftest.png")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.out)
    else:
        ap.error("provide --selftest, or import render_alignment() from a dataset loop")
