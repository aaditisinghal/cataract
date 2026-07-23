"""Attack v0: the dumbest decoder that could work (MASTER_PLAN S5).

Maps the image-patch grid (n_patches, d) -> a reconstructed page (3, H, W). Resist starting with
diffusion; this convolutional decoder is enough to get a kill-test signal in week one.

The one non-obvious trick is the **text-region-weighted loss**. Plain L1/LPIPS optimize average pixel
error, and the lowest-energy way to reduce that on a document is to blur the text into a gray smear.
Up-weighting loss inside annotated field boxes (5-10x) forces the decoder to spend capacity making
text *legible* — which is what PFRR measures — rather than making the page look nice.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchguard.data.align import to_resized_coords
from patchguard.data.fields import Box


class PatchGridDecoder(nn.Module):
    """(B, n_patches, d) image-patch grid -> (B, 3, H, W) reconstruction, in [0, 1].

    Reshapes the row-major grid to (B, d, gh, gw), upsamples by conv blocks to >= target, then
    resizes exactly to ``out_size``. Works for any grid/target (tested small on CPU, runs 32x32->448
    on GPU).
    """

    def __init__(
        self,
        dim: int = 128,
        grid: tuple[int, int] = (32, 32),
        out_size: tuple[int, int] = (448, 448),
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.out_size = out_size  # (H, W)

        gh, gw = grid
        oh, ow = out_size
        self.proj = nn.Conv2d(dim, base_channels, kernel_size=1)

        # Number of 2x upsamples needed to reach or exceed the target from the grid.
        import math

        n_up = max(1, math.ceil(math.log2(max(oh / gh, ow / gw))))
        blocks: list[nn.Module] = []
        c = base_channels
        for _ in range(n_up):
            nxt = max(32, c // 2)
            blocks += [
                nn.ConvTranspose2d(c, nxt, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, nxt),
                nn.GELU(),
            ]
            c = nxt
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        gh, gw = self.grid
        if n < gh * gw:
            raise ValueError(f"got {n} patches, need >= {gh * gw} for grid {self.grid}")
        if d != self.dim:
            raise ValueError(f"patch dim {d} != decoder dim {self.dim}")
        # Take the image-patch block, reshape row-major to a spatial map.
        x = patches[:, : gh * gw, :].transpose(1, 2).reshape(b, d, gh, gw)
        x = self.proj(x)
        x = self.up(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x))


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-pixel weighted L1: sum(w * |pred-target|) / sum(w). weight broadcast over channels."""
    err = (pred - target).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
    denom = weight.sum().clamp_min(1e-8)
    return (weight * err).sum() / denom


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    field_weight: torch.Tensor | None = None,
    lpips_fn: nn.Module | None = None,
    w_l1: float = 1.0,
    w_field: float = 5.0,
    w_lpips: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined attack loss: L1 + (field-weighted L1) + optional LPIPS.

    ``field_weight`` is a (B,1,H,W) map that is >1 inside PII field boxes (see
    ``pixel_weight_from_fields``). ``lpips_fn`` is optional so the module has no hard LPIPS dep.
    Returns (total, components) where components are plain floats for logging.
    """
    l1 = F.l1_loss(pred, target)
    total = w_l1 * l1
    comps = {"l1": float(l1.detach())}

    if field_weight is not None and w_field > 0:
        fw = weighted_l1(pred, target, field_weight)
        total = total + w_field * fw
        comps["field_l1"] = float(fw.detach())

    if lpips_fn is not None and w_lpips > 0:
        # LPIPS expects inputs in [-1, 1].
        lp = lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
        total = total + w_lpips * lp
        comps["lpips"] = float(lp.detach())

    comps["total"] = float(total.detach())
    return total, comps


def ink_weight_map(target: torch.Tensor, ink_boost: float = 15.0) -> torch.Tensor:
    """Per-pixel loss weight from target darkness (1 - luminance).

    The fix for white-collapse: documents are ~90% white, so unweighted losses reward a blank page.
    Weighting by how dark each target pixel is forces the attacker to reproduce INK, not paper.
    target: (..., 3, H, W) in [0,1] -> (..., 1, H, W).
    """
    lum = target.mean(dim=-3, keepdim=True)
    return ink_boost * (1.0 - lum).clamp(0.0, 1.0)


def pixel_weight_from_fields(
    fields_boxes: list[Box],
    orig_size: tuple[int, int],
    out_size: tuple[int, int],
    resize_policy: str = "squash",
    inside_weight: float = 8.0,
    outside_weight: float = 1.0,
) -> torch.Tensor:
    """Build a (1,1,H,W) loss-weight map: ``inside_weight`` within field boxes, else outside.

    Boxes are given in ORIGINAL pixels and mapped through the model's resize policy to the
    reconstruction canvas (``out_size`` = (H, W)), so weighting lines up with where PII actually is.
    """
    oh, ow = out_size
    w = np.full((oh, ow), outside_weight, dtype=np.float32)
    for box in fields_boxes:
        rx0, ry0, rx1, ry1 = to_resized_coords(box, orig_size, (ow, oh), resize_policy)
        x0, y0 = max(0, int(np.floor(rx0))), max(0, int(np.floor(ry0)))
        x1, y1 = min(ow, int(np.ceil(rx1))), min(oh, int(np.ceil(ry1)))
        if x1 > x0 and y1 > y0:
            w[y0:y1, x0:x1] = inside_weight
    return torch.from_numpy(w)[None, None]  # (1,1,H,W)
