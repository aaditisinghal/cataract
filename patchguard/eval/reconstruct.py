"""Reconstruct -> OCR -> PFRR (MASTER_PLAN S4-real). The measurement that decides the thesis.

Loss going down proves the decoder learns; THIS proves whether the attacker actually recovers the
account number. Pipeline per page:
  1. invert the (optionally defended) patch grid with a trained decoder -> reconstructed page
  2. for each annotated field, crop its region from the reconstruction (mapped through the model's
     resize policy) and OCR it
  3. exact-match the OCR text against ground truth -> FieldResult (raw + normalized), aggregate = PFRR

OCR engines are pluggable and heavy imports are deferred, so the pure crop/match logic is CPU-testable
with a MockOCR. Report TWO real engines (Tesseract + PaddleOCR) in the paper — a result that holds
under only one OCR engine isn't a result.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from patchguard.data.align import to_resized_coords
from patchguard.data.fields import AnnotatedField
from patchguard.eval.pfrr import FieldResult, field_recovery
from patchguard.retrievers.base import PageEncoding


class OCREngine(Protocol):
    def read(self, image_crop: np.ndarray) -> str: ...


def reconstruct(decoder: object, encoding: PageEncoding, device: str = "cpu") -> np.ndarray:
    """Invert an encoding's image-patch grid -> (H, W, 3) uint8 page. `decoder` is a PatchGridDecoder."""
    import torch

    patches = torch.from_numpy(encoding.image_patches()).float()[None].to(device)
    decoder.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        out = decoder(patches)  # type: ignore[operator]  (B,3,H,W) in [0,1]
    img = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def ocr_field_pfrr(
    recon_image: np.ndarray,
    fields: list[AnnotatedField],
    orig_size: tuple[int, int],
    out_size: tuple[int, int],
    resize_policy: str,
    ocr: OCREngine,
    pad: int = 2,
) -> list[FieldResult]:
    """Crop each field from the reconstruction, OCR it, and score against ground truth.

    ``orig_size`` = (w,h) of the source page (field boxes are in these px). ``out_size`` = (H,W) of the
    reconstruction canvas. Boxes are mapped through the SAME resize policy the model used.
    """
    oh, ow = out_size
    results: list[FieldResult] = []
    for f in fields:
        # map original-px box -> reconstruction canvas px (input_size passed as (width,height))
        rx0, ry0, rx1, ry1 = to_resized_coords(f.box, orig_size, (ow, oh), resize_policy)
        x0 = max(0, int(np.floor(rx0)) - pad)
        y0 = max(0, int(np.floor(ry0)) - pad)
        x1 = min(ow, int(np.ceil(rx1)) + pad)
        y1 = min(oh, int(np.ceil(ry1)) + pad)
        crop = recon_image[y0:y1, x0:x1]
        text = ocr.read(_upscale_for_ocr(crop)) if crop.size > 0 else ""
        results.append(field_recovery(f.field_type, f.text, text))
    return results


def _upscale_for_ocr(crop: np.ndarray, min_height: int = 40) -> np.ndarray:
    """Tesseract needs ~30-40px cap height. Field crops off a 448px canvas are tiny, so upscale.

    Does not add detail — just gives OCR a fighting chance on small-but-legible text.
    """
    h = crop.shape[0]
    if h == 0 or h >= min_height:
        return crop
    try:
        from PIL import Image

        scale = int(np.ceil(min_height / h))
        im = Image.fromarray(crop)
        return np.array(im.resize((crop.shape[1] * scale, h * scale), Image.LANCZOS))
    except Exception:
        return crop


class TesseractOCR:
    """Deployable OCR engine #1. Requires pytesseract + the tesseract binary (in the repro image)."""

    def read(self, image_crop: np.ndarray) -> str:
        import pytesseract
        from PIL import Image

        if image_crop.size == 0:
            return ""
        return pytesseract.image_to_string(Image.fromarray(image_crop)).strip()


class PaddleOCREngine:
    """OCR engine #2 (cross-check). Requires paddleocr. Lazy-initialized."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)

    def read(self, image_crop: np.ndarray) -> str:
        if image_crop.size == 0:
            return ""
        res = self._ocr.ocr(image_crop, cls=False)
        if not res or not res[0]:
            return ""
        return " ".join(line[1][0] for line in res[0]).strip()
