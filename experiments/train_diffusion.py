"""Train + eyeball the diffusion (generative-prior) attack on FUNSD (MASTER_PLAN S6).

Trains the patch-grid -> SD-VAE-latent projector with adversarial + perceptual + text-weighted losses,
then dumps original|reconstruction images at 512px so we can see whether text is now LEGIBLE (the
question attack v0 answered "no" to). GCS-aware like train_funsd.

    python -m experiments.train_diffusion --data gs://.../funsd --out gs://.../runs/diffusion-<sha>
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from patchguard.attack.diffusion import (
    DiffusionInverter,
    DiffusionTrainConfig,
    PatchDiscriminator,
    train_diffusion,
)
from patchguard.attack.train import TrainConfig, build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/diffusion")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--limit", type=int, default=149)
    ap.add_argument("--dump", type=int, default=6, help="how many recon images to save")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from patchguard.attack.diffusion import DiffusersVAEAdapter
    from patchguard.eval.reconstruct import reconstruct
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_download, _gcs_upload, load_funsd_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_size = (512, 512)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    (local_out / "recon_samples").mkdir(parents=True, exist_ok=True)

    data_root = args.data
    if str(data_root).startswith("gs://"):
        data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))
    samples = load_funsd_samples(data_root, "training_data", args.limit, "word")
    print(f"train samples: {len(samples)}")

    retriever = ColPaliRetriever(model_name=args.model)
    vae = DiffusersVAEAdapter(model=args.vae, device=device, dtype="float32")
    # ink_boost weights the loss by pixel darkness -> forces ink reproduction, not a blank page
    ds = build_dataset(
        samples, retriever, TrainConfig(out_size=out_size, inside_weight=10.0, ink_boost=20.0)
    )
    inverter = DiffusionInverter(dim=ds.dim, grid=ds.grid, vae=vae, latent_size=(64, 64))
    disc = PatchDiscriminator()
    cfg = DiffusionTrainConfig(out_size=out_size, epochs=args.epochs, ckpt_dir=str(local_out))
    out = train_diffusion(ds, inverter, disc, cfg)
    print(f"trained diffusion inverter -> {out['checkpoint']}")
    print(f"final: {out['history'][-1]}")

    # eyeball: original | reconstruction at 512px
    inverter.eval()
    for i in range(min(args.dump, len(samples))):
        enc = retriever.encode_page(samples[i].image)
        recon = reconstruct(inverter, enc, device)
        orig = np.array(Image.fromarray(samples[i].image).resize((out_size[1], out_size[0])))
        Image.fromarray(np.concatenate([orig, recon], axis=1)).save(
            local_out / "recon_samples" / f"recon_{i}.png"
        )
    print(f"dumped {min(args.dump, len(samples))} recon images")

    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
        print(f"uploaded -> {args.out}")


if __name__ == "__main__":
    main()
