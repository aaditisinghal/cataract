"""Decoder training loop (MASTER_PLAN S5/S6).

Trains PatchGridDecoder to invert a retriever's image-patch grid back to the page. Structured so the
GPU-heavy parts (encoding, training) sit behind the Retriever interface and a plain torch loop, and
the pure glue (dataset assembly, field weighting, checkpoint/metrics) is CPU-testable.

Key choice: the dataset stores each page's **image_patches** (the gh*gw grid the attacker actually
inverts), not the full token sequence — so the decoder input lines up with its grid regardless of
prefix/trailing tokens. Targets are the page resized to the reconstruction canvas; the field-weight
map upweights PII regions (attack/decoder.pixel_weight_from_fields).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from patchguard.attack.decoder import (
    PatchGridDecoder,
    ink_weight_map,
    pixel_weight_from_fields,
    reconstruction_loss,
)
from patchguard.data.fields import Box
from patchguard.repro import run_fingerprint, seed_everything
from patchguard.retrievers.base import PageEncoding, Retriever


@dataclass
class TrainConfig:
    out_size: tuple[int, int] = (448, 448)
    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-4
    base_channels: int = 256
    w_l1: float = 1.0
    w_field: float = 5.0
    w_lpips: float = 1.0
    inside_weight: float = 8.0
    ink_boost: float = 0.0  # >0 adds darkness-weighted loss (fixes white-collapse on documents)
    seed: int = 0
    device: str | None = None
    ckpt_dir: str = "results/decoder"
    use_lpips: bool = True


@dataclass
class Sample:
    image: np.ndarray  # (H, W, 3)
    field_boxes: list[Box]
    orig_size: tuple[int, int]  # (width, height)


class PatchReconDataset(Dataset):
    """In-memory (image_patches -> target image, field_weight) triples for decoder training."""

    def __init__(
        self,
        encodings: Sequence[PageEncoding],
        images: Sequence[np.ndarray],
        field_boxes: Sequence[list[Box]],
        orig_sizes: Sequence[tuple[int, int]],
        out_size: tuple[int, int],
        resize_policy: str,
        inside_weight: float,
        ink_boost: float = 0.0,
    ) -> None:
        if not encodings:
            raise ValueError("empty dataset")
        self.out_size = out_size
        self.grid = encodings[0].grid
        self.dim = encodings[0].dim
        self._patches = [torch.from_numpy(e.image_patches()).float() for e in encodings]
        self._targets = [_image_to_target(im, out_size) for im in images]
        self._weights = []
        for fb, os_, tgt in zip(field_boxes, orig_sizes, self._targets):
            w = pixel_weight_from_fields(fb, os_, out_size, resize_policy, inside_weight)[0]
            if ink_boost > 0:
                w = w + ink_weight_map(tgt, ink_boost)  # amplify ink pixels
            self._weights.append(w)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._patches[i], self._targets[i], self._weights[i]


def build_dataset(
    samples: Sequence[Sample], retriever: Retriever, cfg: TrainConfig
) -> PatchReconDataset:
    """Encode each sample once (cache the grid) and assemble the training tensors."""
    encs = [retriever.encode_page(s.image) for s in samples]
    return PatchReconDataset(
        encodings=encs,
        images=[s.image for s in samples],
        field_boxes=[s.field_boxes for s in samples],
        orig_sizes=[s.orig_size for s in samples],
        out_size=cfg.out_size,
        resize_policy=encs[0].resize_policy,
        inside_weight=cfg.inside_weight,
        ink_boost=cfg.ink_boost,
    )


def train_decoder(dataset: PatchReconDataset, cfg: TrainConfig) -> dict[str, object]:
    """Train the decoder; checkpoint + metrics.json (with fingerprint) to cfg.ckpt_dir."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dec = PatchGridDecoder(
        dim=dataset.dim, grid=dataset.grid, out_size=cfg.out_size, base_channels=cfg.base_channels
    ).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=cfg.lr)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    lpips_fn = _maybe_lpips(cfg, device)

    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        dec.train()
        agg: dict[str, float] = {}
        n = 0
        for patches, target, weight in loader:
            patches, target, weight = patches.to(device), target.to(device), weight.to(device)
            pred = dec(patches)
            loss, comps = reconstruction_loss(
                pred, target, field_weight=weight, lpips_fn=lpips_fn,
                w_l1=cfg.w_l1, w_field=cfg.w_field, w_lpips=cfg.w_lpips,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in comps.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        history.append({"epoch": epoch, **{k: v / max(n, 1) for k, v in agg.items()}})

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "decoder.pt"
    torch.save({"state_dict": dec.state_dict(), "config": asdict(cfg), "grid": dataset.grid}, ckpt_path)
    metrics = {
        "config": asdict(cfg),
        "n_samples": len(dataset),
        "final": history[-1] if history else {},
        "history": history,
        "fingerprint": run_fingerprint(),
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"checkpoint": str(ckpt_path), "history": history, "metrics": metrics}


def _image_to_target(image: np.ndarray, out_size: tuple[int, int]) -> torch.Tensor:
    """(H,W,3) uint8/float -> (3, OH, OW) float in [0,1], bilinear-resized."""
    t = torch.from_numpy(np.asarray(image)).float()
    if t.max() > 1.5:
        t = t / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = torch.nn.functional.interpolate(t, size=out_size, mode="bilinear", align_corners=False)
    return t[0]


def _maybe_lpips(cfg: TrainConfig, device: torch.device) -> torch.nn.Module | None:
    if not cfg.use_lpips or cfg.w_lpips <= 0:
        return None
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception:
        return None  # LPIPS optional; L1 + field-weighted L1 still train a legible decoder
