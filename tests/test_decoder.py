"""Attack-decoder tests on CPU (small config); shapes, backprop, and the text-region weighting."""

import numpy as np
import torch

from patchguard.attack.decoder import (
    PatchGridDecoder,
    pixel_weight_from_fields,
    reconstruction_loss,
    weighted_l1,
)


def test_decoder_output_shape_small():
    dec = PatchGridDecoder(dim=8, grid=(4, 4), out_size=(32, 32), base_channels=32)
    x = torch.randn(2, 16, 8)  # (B, n_patches, d)
    with torch.no_grad():
        y = dec(x)
    assert y.shape == (2, 3, 32, 32)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0  # sigmoid range


def test_decoder_handles_trailing_tokens():
    # 16 image patches + 3 trailing instruction tokens; decoder must use only the first 16.
    dec = PatchGridDecoder(dim=8, grid=(4, 4), out_size=(16, 16), base_channels=16)
    x = torch.randn(1, 19, 8)
    assert dec(x).shape == (1, 3, 16, 16)


def test_decoder_colpali_shape():
    dec = PatchGridDecoder(dim=128, grid=(32, 32), out_size=(448, 448))
    x = torch.randn(1, 1030, 128)  # ColPali: 1024 image + 6 instruction tokens
    assert dec(x).shape == (1, 3, 448, 448)


def test_backprop_flows():
    dec = PatchGridDecoder(dim=8, grid=(4, 4), out_size=(32, 32), base_channels=16)
    x = torch.randn(2, 16, 8, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    fw = torch.ones(1, 1, 32, 32)
    loss, comps = reconstruction_loss(dec(x), target, field_weight=fw, w_field=5.0)
    loss.backward()
    assert np.isfinite(comps["total"])
    assert any(p.grad is not None for p in dec.parameters())


def test_weighted_l1_upweights_flagged_region():
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    target[..., 0, 0] = 1.0  # one wrong pixel at (0,0)
    heavy = torch.ones(1, 1, 4, 4)
    heavy[..., 0, 0] = 10.0
    flat = weighted_l1(pred, target, torch.ones(1, 1, 4, 4))
    focused = weighted_l1(pred, target, heavy)
    # concentrating weight on the erroneous pixel raises its contribution
    assert float(focused) > float(flat)


def test_pixel_weight_from_fields_marks_boxes():
    # one box covering the top-left quadrant of a 32x32 canvas (no scaling)
    w = pixel_weight_from_fields(
        [(0, 0, 16, 16)], orig_size=(32, 32), out_size=(32, 32), inside_weight=8.0
    )
    assert w.shape == (1, 1, 32, 32)
    assert float(w[0, 0, 4, 4]) == 8.0  # inside box
    assert float(w[0, 0, 24, 24]) == 1.0  # outside box


def test_pixel_weight_respects_squash_scaling():
    # original 64 wide -> 32 canvas: box x in [0,32] original maps to [0,16] on canvas
    w = pixel_weight_from_fields(
        [(0, 0, 32, 64)], orig_size=(64, 64), out_size=(32, 32), inside_weight=5.0
    )
    assert float(w[0, 0, 10, 8]) == 5.0  # inside mapped box (x<16)
    assert float(w[0, 0, 10, 24]) == 1.0  # outside (x>16)
