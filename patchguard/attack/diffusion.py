"""Attack v1: the generative-prior (diffusion) attack (MASTER_PLAN S6).

Why not literally condition Stable Diffusion's UNet: SD's prior is trained on natural images and
hallucinates gibberish for document *text*. The recipe that actually recovers legible text is what
attack v0 lacked — a strong image prior + an anti-blur objective:

  patch grid --project--> SD frozen-VAE latent (4x64x64) --frozen VAE decode--> 512x512 page

trained with: latent-L2 (match the target page's VAE latent) + pixel L1 + field-weighted L1 + LPIPS
+ **adversarial** (a PatchGAN discriminator). L1/L2 alone give the blurry mean that killed v0; the
discriminator pushes toward sharp, realistic strokes — i.e. readable characters.

The VAE is pluggable (VAEBackend protocol) so the heavy Stable-Diffusion autoencoder loads only in
the container; CPU tests use a tiny stand-in. Everything else here is a plain nn.Module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEBackend(Protocol):
    """Maps images in [0,1] <-> latents. Real impl wraps diffusers AutoencoderKL; tests use a tiny one."""

    def encode_latent(self, image_bchw: torch.Tensor) -> torch.Tensor: ...
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor: ...


class LatentProjector(nn.Module):
    """(B, n_patches, d) image-patch grid -> (B, 4, Hl, Wl) VAE latent.

    Channel-projects then resamples the gh*gw grid to the latent size, so it handles any grid
    (ColPali 32x32 AND pooled-BiPali 1x1) into the same latent shape.
    """

    def __init__(
        self, dim: int, grid: tuple[int, int], latent_size: tuple[int, int] = (64, 64),
        latent_ch: int = 4, hidden: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.latent_size = latent_size
        self.proj = nn.Conv2d(dim, hidden, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GroupNorm(8, hidden // 2), nn.GELU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden // 2, latent_ch, 3, padding=1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        gh, gw = self.grid
        x = patches[:, : gh * gw, :].transpose(1, 2).reshape(b, d, gh, gw)
        x = self.proj(x)
        x = F.interpolate(x, size=self.latent_size, mode="bilinear", align_corners=False)
        x = self.refine(x)
        return self.head(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator -> per-patch realness map. The anti-blur signal."""

    def __init__(self, in_ch: int = 3, base: int = 64) -> None:
        super().__init__()

        def block(i: int, o: int, s: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(i, o, 4, stride=s, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, 2, norm=False),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            *block(base * 4, base * 8, 1),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionInverter(nn.Module):
    """Projector + frozen VAE decoder. forward(patches) -> reconstructed page in [0,1]."""

    def __init__(self, dim: int, grid: tuple[int, int], vae: VAEBackend,
                 latent_size: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.projector = LatentProjector(dim, grid, latent_size=latent_size)
        self.vae = vae

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        return self.projector(patches)

    @torch.no_grad()
    def encode_target(self, image_bchw: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_latent(image_bchw)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode_latent(latent).clamp(0, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decode(self.project(patches))


# ---- adversarial (hinge) losses -------------------------------------------------------------------

def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def latent_l2(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_latent, target_latent)


# ---- real Stable-Diffusion VAE adapter (loads only in the container) -------------------------------

class DiffusersVAEAdapter:
    """Wraps a frozen diffusers AutoencoderKL to the VAEBackend interface (images in [0,1])."""

    def __init__(self, model: str = "stabilityai/sd-vae-ft-mse", device: str = "cuda",
                 dtype: str = "float32", scale: float = 0.18215) -> None:
        from diffusers import AutoencoderKL

        self.scale = scale
        td = getattr(torch, dtype)
        self.vae = AutoencoderKL.from_pretrained(model, torch_dtype=td).to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def encode_latent(self, image_bchw: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(image_bchw * 2 - 1).latent_dist
        return posterior.mean * self.scale

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(latent / self.scale).sample
        return (img + 1) / 2


# ---- GAN training loop ----------------------------------------------------------------------------

@dataclass
class DiffusionTrainConfig:
    out_size: tuple[int, int] = (512, 512)
    epochs: int = 60
    batch_size: int = 4
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    w_latent: float = 1.0
    w_l1: float = 1.0
    w_field: float = 5.0
    w_lpips: float = 1.0
    w_adv: float = 0.1
    adv_warmup_epochs: int = 3  # let the projector find structure before the discriminator kicks in
    seed: int = 0
    device: str | None = None
    ckpt_dir: str = "results/diffusion"
    use_lpips: bool = True


def train_diffusion(
    dataset, inverter: DiffusionInverter, discriminator: PatchDiscriminator, cfg: DiffusionTrainConfig
) -> dict[str, object]:
    """Alternating G/D training. Only the projector (+ discriminator) learn; the VAE stays frozen.

    ``dataset`` must yield (image_patches, target_image@out_size, field_weight@out_size) — build it
    with attack.train.build_dataset using out_size == cfg.out_size.
    """
    from torch.utils.data import DataLoader

    from patchguard.attack.decoder import reconstruction_loss
    from patchguard.repro import run_fingerprint, seed_everything

    seed_everything(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inverter.to(device)
    discriminator.to(device)
    optG = torch.optim.Adam(inverter.projector.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    lpips_fn = _maybe_lpips(cfg, device)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        adv_on = epoch >= cfg.adv_warmup_epochs
        agg: dict[str, float] = {}
        n = 0
        for patches, target, weight in loader:
            patches, target, weight = patches.to(device), target.to(device), weight.to(device)
            target_latent = inverter.encode_target(target)
            pred_latent = inverter.project(patches)
            fake = inverter.decode(pred_latent)

            if adv_on:
                optD.zero_grad()
                d_loss = d_hinge_loss(discriminator(target), discriminator(fake.detach()))
                d_loss.backward()
                optD.step()
            else:
                d_loss = torch.tensor(0.0)

            optG.zero_grad()
            l_lat = latent_l2(pred_latent, target_latent)
            pix, comps = reconstruction_loss(
                fake, target, field_weight=weight, lpips_fn=lpips_fn,
                w_l1=cfg.w_l1, w_field=cfg.w_field, w_lpips=cfg.w_lpips,
            )
            g_adv = g_hinge_loss(discriminator(fake)) if adv_on else torch.tensor(0.0, device=device)
            g_loss = cfg.w_latent * l_lat + pix + (cfg.w_adv * g_adv if adv_on else 0.0)
            g_loss.backward()
            optG.step()

            for k, v in comps.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["latent"] = agg.get("latent", 0.0) + float(l_lat.detach())
            agg["g_adv"] = agg.get("g_adv", 0.0) + float(g_adv.detach())
            agg["d"] = agg.get("d", 0.0) + float(d_loss.detach())
            n += 1
        history.append({"epoch": epoch, "adv_on": float(adv_on), **{k: v / max(n, 1) for k, v in agg.items()}})

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "inverter.pt"
    torch.save({"projector": inverter.projector.state_dict(), "config": asdict(cfg),
                "grid": inverter.projector.grid, "dim": inverter.projector.dim}, ckpt)
    (ckpt_dir / "metrics.json").write_text(json.dumps(
        {"config": asdict(cfg), "n_samples": len(dataset), "final": history[-1] if history else {},
         "history": history, "fingerprint": run_fingerprint()}, indent=2))
    return {"checkpoint": str(ckpt), "history": history}


def _maybe_lpips(cfg: DiffusionTrainConfig, device: torch.device) -> nn.Module | None:
    if not cfg.use_lpips or cfg.w_lpips <= 0:
        return None
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        return None
