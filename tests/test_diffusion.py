"""CPU tests for the diffusion (generative-prior) attack, using a tiny stand-in VAE."""

import numpy as np
import torch
import torch.nn as nn

from patchguard.attack.decoder import PatchGridDecoder  # noqa: F401 (ensures module import path)
from patchguard.attack.diffusion import (
    DiffusionInverter,
    DiffusionTrainConfig,
    LatentProjector,
    PatchDiscriminator,
    d_hinge_loss,
    g_hinge_loss,
    latent_l2,
    train_diffusion,
)
from patchguard.attack.train import Sample, TrainConfig, build_dataset
from patchguard.retrievers.mock import MockRetriever


class TinyVAE:
    """3x(HxW) <-> 4x(H/8 x W/8). Plain object (not a submodule) so its params never enter optG."""

    def __init__(self):
        self.enc = nn.Conv2d(3, 4, 8, stride=8)
        self.dec = nn.ConvTranspose2d(4, 3, 8, stride=8)
        for p in list(self.enc.parameters()) + list(self.dec.parameters()):
            p.requires_grad_(False)

    def encode_latent(self, x):
        return self.enc(x)

    def decode_latent(self, z):
        return self.dec(z).sigmoid()


def test_latent_projector_colpali_and_bipali_grids():
    # ColPali-like grid
    proj = LatentProjector(dim=8, grid=(4, 4), latent_size=(8, 8))
    assert proj(torch.randn(2, 16, 8)).shape == (2, 4, 8, 8)
    # BiPali 1x1 grid -> same latent shape (single vector -> full latent)
    projb = LatentProjector(dim=8, grid=(1, 1), latent_size=(8, 8))
    assert projb(torch.randn(2, 1, 8)).shape == (2, 4, 8, 8)


def test_patch_discriminator_shape():
    d = PatchDiscriminator()
    out = d(torch.randn(2, 3, 64, 64))
    assert out.ndim == 4 and out.shape[0] == 2 and out.shape[2] > 0


def test_inverter_forward_shape_and_range():
    inv = DiffusionInverter(dim=8, grid=(4, 4), vae=TinyVAE(), latent_size=(8, 8))
    with torch.no_grad():
        img = inv(torch.randn(2, 16, 8))
    assert img.shape == (2, 3, 64, 64)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_hinge_losses_signs():
    real = torch.full((2, 1, 4, 4), 2.0)
    fake = torch.full((2, 1, 4, 4), -2.0)
    # discriminator confidently correct -> ~0 loss; generator wants fake_logits high -> positive loss
    assert float(d_hinge_loss(real, fake)) < 1e-4
    assert float(g_hinge_loss(fake)) == 2.0
    assert float(latent_l2(torch.zeros(2, 4, 4, 4), torch.zeros(2, 4, 4, 4))) == 0.0


def _samples(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        img[8:20, 8:48] = 30
        out.append(Sample(image=img, field_boxes=[(8, 8, 48, 20)], orig_size=(64, 64)))
    return out


def test_train_diffusion_runs_and_checkpoints(tmp_path):
    r = MockRetriever(grid=(4, 4), input_size=(64, 64), dim=8, n_prefix_tokens=1)
    ds = build_dataset(_samples(6), r, TrainConfig(out_size=(64, 64), inside_weight=8.0))
    inv = DiffusionInverter(dim=ds.dim, grid=ds.grid, vae=TinyVAE(), latent_size=(8, 8))
    disc = PatchDiscriminator()
    cfg = DiffusionTrainConfig(out_size=(64, 64), epochs=4, batch_size=3, adv_warmup_epochs=1,
                               use_lpips=False, w_lpips=0.0, ckpt_dir=str(tmp_path))
    out = train_diffusion(ds, inv, disc, cfg)
    assert len(out["history"]) == 4
    assert np.isfinite(out["history"][-1]["total"])
    assert (tmp_path / "inverter.pt").exists()
