"""Inversion attacks (MASTER_PLAN S5, S6, S9)."""

from patchguard.attack.decoder import (
    PatchGridDecoder,
    pixel_weight_from_fields,
    reconstruction_loss,
    weighted_l1,
)
from patchguard.attack.diffusion import (
    DiffusionInverter,
    LatentProjector,
    PatchDiscriminator,
    train_diffusion,
)

__all__ = [
    "PatchGridDecoder",
    "pixel_weight_from_fields",
    "reconstruction_loss",
    "weighted_l1",
    "DiffusionInverter",
    "LatentProjector",
    "PatchDiscriminator",
    "train_diffusion",
]
