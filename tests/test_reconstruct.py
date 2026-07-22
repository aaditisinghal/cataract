"""CPU test of reconstruct -> crop -> OCR -> PFRR, using a mock OCR (no tesseract needed)."""

import numpy as np

from patchguard.attack.decoder import PatchGridDecoder
from patchguard.data.fields import AnnotatedField
from patchguard.eval.pfrr import pfrr
from patchguard.eval.reconstruct import ocr_field_pfrr, reconstruct
from patchguard.retrievers.mock import MockRetriever


class QueueOCR:
    """Returns preset strings in order, ignoring the pixel crop — lets us control recovery in tests."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0
        self.crop_shapes = []

    def read(self, crop):
        self.crop_shapes.append(crop.shape)
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


def test_reconstruct_shape_and_dtype():
    dec = PatchGridDecoder(dim=8, grid=(4, 4), out_size=(32, 32), base_channels=16)
    r = MockRetriever(grid=(4, 4), input_size=(32, 32), dim=8, n_prefix_tokens=1)
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    enc = r.encode_page(img)
    recon = reconstruct(dec, enc, device="cpu")
    assert recon.shape == (32, 32, 3)
    assert recon.dtype == np.uint8


def test_ocr_field_pfrr_recovers_when_ocr_matches():
    recon = np.zeros((32, 32, 3), dtype=np.uint8)
    fields = [
        AnnotatedField("account_no", "4055", (0, 0, 16, 8)),
        AnnotatedField("name", "Smith", (0, 16, 16, 24)),
    ]
    # OCR returns an OCR-confusable version of the account number, and a wrong name.
    ocr = QueueOCR(["4O5S", "Jones"])
    results = ocr_field_pfrr(
        recon, fields, orig_size=(32, 32), out_size=(32, 32), resize_policy="squash", ocr=ocr
    )
    assert len(results) == 2
    acct = next(r for r in results if r.field_type == "account_no")
    name = next(r for r in results if r.field_type == "name")
    assert acct.normalized_exact is True  # 4055 vs 4O5S -> normalized match
    assert name.normalized_exact is False
    # crops were actually taken (non-empty regions)
    assert all(s[0] > 0 and s[1] > 0 for s in ocr.crop_shapes)


def test_pfrr_aggregate_over_reconstructed_fields():
    recon = np.zeros((32, 32, 3), dtype=np.uint8)
    fields = [
        AnnotatedField("account_no", "4055", (0, 0, 16, 8)),
        AnnotatedField("account_no", "1200", (0, 8, 16, 16)),
    ]
    ocr = QueueOCR(["4055", "9999"])  # one hit, one miss
    results = ocr_field_pfrr(recon, fields, (32, 32), (32, 32), "squash", ocr)
    agg = pfrr(results, normalized=True)
    assert agg["account_no"]["n"] == 2
    assert abs(agg["account_no"]["recovery_rate"] - 0.5) < 1e-9


def test_empty_crop_yields_empty_ocr():
    recon = np.zeros((32, 32, 3), dtype=np.uint8)
    # degenerate/out-of-canvas box -> empty crop -> empty string -> miss
    fields = [AnnotatedField("id_no", "ABC123", (40, 40, 45, 45))]
    ocr = QueueOCR(["should-not-be-used"])
    results = ocr_field_pfrr(recon, fields, (32, 32), (32, 32), "squash", ocr)
    assert results[0].normalized_exact is False
