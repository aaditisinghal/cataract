"""Overfit probe — the decisive info-presence test (MASTER_PLAN diagnostic).

The core ambiguity: does ColPali retain glyph-level info (attack too weak = fixable) or not
(fundamental = structure-leakage paper)? This isolates it by giving the attack its BEST possible shot
and removing the two confounds:

  * removes generalization: train AND evaluate PFRR on the SAME pages (memorization test).
  * removes the SD-VAE bottleneck: a high-capacity PIXEL decoder at high resolution.
  * removes white-collapse: ink-weighted loss.

Read: if train-set PFRR is still ~0 -> the model cannot reproduce text it was DIRECTLY optimized on
-> the glyph info is not recoverable (absent or sub-resolution) -> commit to the structure paper.
If train-set PFRR is high but held-out was ~0 -> info IS present, it's a generalization/capacity gap
-> the attack is worth scaling. Either way it kills the ambiguity.

GCS-aware like the other entrypoints.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/overfit")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=8, help="pages to overfit (small = strongest memorization)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--max-fields", type=int, default=40)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from patchguard.attack.decoder import PatchGridDecoder
    from patchguard.attack.train import Sample, TrainConfig, build_dataset, train_decoder
    from patchguard.data.funsd import iter_funsd
    from patchguard.eval.pfrr import pfrr
    from patchguard.eval.reconstruct import TesseractOCR, ocr_field_pfrr, reconstruct
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_download, _gcs_upload

    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = (args.resolution, args.resolution)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    (local_out / "recon_samples").mkdir(parents=True, exist_ok=True)

    data_root = args.data
    if str(data_root).startswith("gs://"):
        data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))

    pages = []  # (image, fields, size)
    for ps in iter_funsd(data_root, split="training_data", granularity="word"):
        pages.append((np.array(Image.open(ps.image_path).convert("RGB")), ps.fields, ps.size))
        if len(pages) >= args.n:
            break
    print(f"overfitting on {len(pages)} pages @ {args.resolution}px, {args.channels}ch, {args.epochs} epochs")

    retriever = ColPaliRetriever(model_name=args.model)
    samples = [Sample(image=im, field_boxes=[f.box for f in fs], orig_size=sz) for im, fs, sz in pages]
    cfg = TrainConfig(out_size=res, epochs=args.epochs, inside_weight=10.0, ink_boost=20.0,
                      base_channels=args.channels, use_lpips=True, ckpt_dir=str(local_out))
    ds = build_dataset(samples, retriever, cfg)
    train_decoder(ds, cfg)

    dec = PatchGridDecoder(dim=ds.dim, grid=ds.grid, out_size=res, base_channels=args.channels).to(device)
    dec.load_state_dict(torch.load(local_out / "decoder.pt")["state_dict"])

    # PFRR on the SAME training pages (memorization)
    ocr = TesseractOCR()
    all_results = []
    per_page = []
    for i, (im, fs, sz) in enumerate(pages):
        enc = retriever.encode_page(im)
        recon = reconstruct(dec, enc, device)
        r = ocr_field_pfrr(recon, fs[: args.max_fields], sz, res, enc.resize_policy, ocr)
        all_results.extend(r)
        hits = sum(x.normalized_exact for x in r)
        per_page.append(hits / max(len(r), 1))
        orig = np.array(Image.fromarray(im).resize((res[1], res[0])))
        Image.fromarray(np.concatenate([orig, recon], axis=1)).save(
            local_out / "recon_samples" / f"recon_{i}.png"
        )

    agg = pfrr(all_results, normalized=True)
    train_pfrr = float(np.mean(per_page))
    print(f"TRAIN-SET PFRR (memorization) = {train_pfrr:.3f}")
    print(f"per field type: {agg}")

    payload = {
        "mode": "overfit_probe", "n_pages": len(pages), "resolution": args.resolution,
        "epochs": args.epochs, "train_pfrr": train_pfrr, "per_page_pfrr": per_page,
        "by_field_type": agg, "fingerprint": run_fingerprint(),
    }
    (local_out / "overfit.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote overfit.json -> {args.out}")


if __name__ == "__main__":
    main()
