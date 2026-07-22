"""PII localizers (MASTER_PLAN S3b/S8) — which patches hold sensitive fields.

Three rungs, each a row in the localization ablation. The oracle→deployable gap is the honest cost:
  * ``OracleLocalizer``   — from ground-truth field boxes. Upper bound; the only one implemented here
                            (pure geometry, testable offline).
  * ``DetectorLocalizer`` — a layout/PII detector trained on DocLayNet. Realistic. (S8, needs a model.)
  * ``OCRNERLocalizer``   — OCR + NER (Presidio-style) on the reconstructed/rendered page. Deployable
                            today. (S8, needs OCR+NER.)

A localizer returns a full-sequence boolean mask aligned to ``PageEncoding.patches`` (same convention
as align.boxes_to_patch_mask), ready to hand to defense.perturb.patch_scoped_gaussian.
"""

from __future__ import annotations

import numpy as np

from patchguard.data.align import boxes_to_patch_mask
from patchguard.data.fields import Box
from patchguard.retrievers.base import PageEncoding


class OracleLocalizer:
    """Ground-truth localizer: mask = patches covered by the given PII field boxes."""

    def __init__(self, coverage_threshold: float = 0.0) -> None:
        self.coverage_threshold = coverage_threshold

    def locate(
        self, enc: PageEncoding, field_boxes: list[Box], orig_size: tuple[int, int]
    ) -> np.ndarray:
        """field_boxes in ORIGINAL pixels; orig_size = (width, height) of that page."""
        return boxes_to_patch_mask(
            boxes=field_boxes,
            orig_size=orig_size,
            grid=enc.grid,
            input_size=enc.input_size,
            resize_policy=enc.resize_policy,
            coverage_threshold=self.coverage_threshold,
            n_prefix_tokens=enc.n_prefix_tokens,
        )


class DetectorLocalizer:
    """Trained layout/PII detector (S8). Placeholder — wired when the detector model lands."""

    def locate(self, enc: PageEncoding, **kwargs: object) -> np.ndarray:  # noqa: D401
        raise NotImplementedError(
            "DetectorLocalizer needs a trained detector (S8). Use OracleLocalizer for the kill test."
        )


class OCRNERLocalizer:
    """OCR + NER localizer (S8), the deployable rung. Placeholder — wired with OCR+Presidio."""

    def locate(self, enc: PageEncoding, **kwargs: object) -> np.ndarray:  # noqa: D401
        raise NotImplementedError(
            "OCRNERLocalizer needs OCR+NER (S8). Use OracleLocalizer for the kill test."
        )
