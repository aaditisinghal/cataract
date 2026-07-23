"""Big-font PII probe (MASTER_PLAN S6 diagnostic) — the make-or-break for the "photographs" thesis.

Generates synthetic large-font ID cards (exact ground-truth PII), encodes them with ColPali, trains a
high-capacity pixel decoder (ink-weighted, 768px, no SD-VAE bottleneck), and reports PFRR on BOTH the
training cards (can it fit big-font text at all?) and held-out cards (does it generalize?).

Decisive read:
  * train & held-out PFRR both high  -> legible PII IS recoverable when text is legible at encode time
    -> thesis alive; FUNSD failure was resolution, not fundamental. Scale up.
  * train PFRR high, held-out ~0      -> info present + fittable but attack doesn't generalize (capacity)
  * train PFRR ~0 (even on clean big-font cards it optimized on) -> glyph info fundamentally not
    retained by ColPali -> commit to the structure-leakage paper with airtight evidence.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bigfont")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--font-size", type=int, default=34)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from patchguard.attack.decoder import PatchGridDecoder
    from patchguard.attack.train import Sample, TrainConfig, build_dataset, train_decoder
    from patchguard.data.synthdoc import generate_id_cards
    from patchguard.eval.pfrr import pfrr
    from patchguard.eval.reconstruct import TesseractOCR, ocr_field_pfrr, reconstruct
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = (args.resolution, args.resolution)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    (local_out / "recon_samples").mkdir(parents=True, exist_ok=True)

    train_cards = generate_id_cards(args.n_train, seed=1, value_font_size=args.font_size)
    test_cards = generate_id_cards(args.n_test, seed=999, value_font_size=args.font_size)
    print(f"generated {len(train_cards)} train / {len(test_cards)} test cards @ font {args.font_size}px")

    retriever = ColPaliRetriever(model_name=args.model)
    train_s = [Sample(image=im, field_boxes=[f.box for f in fs], orig_size=(im.shape[1], im.shape[0]))
               for im, fs in train_cards]
    cfg = TrainConfig(out_size=res, epochs=args.epochs, inside_weight=10.0, ink_boost=20.0,
                      base_channels=args.channels, use_lpips=True, ckpt_dir=str(local_out))
    ds = build_dataset(train_s, retriever, cfg)
    train_decoder(ds, cfg)

    dec = PatchGridDecoder(dim=ds.dim, grid=ds.grid, out_size=res, base_channels=args.channels).to(device)
    dec.load_state_dict(torch.load(local_out / "decoder.pt")["state_dict"])

    ocr = TesseractOCR()

    def eval_cards(cards, tag, dump=0):
        results, per = [], []
        for i, (im, fs) in enumerate(cards):
            enc = retriever.encode_page(im)
            recon = reconstruct(dec, enc, device)
            r = ocr_field_pfrr(recon, fs, (im.shape[1], im.shape[0]), res, enc.resize_policy, ocr)
            results.extend(r)
            per.append(sum(x.normalized_exact for x in r) / max(len(r), 1))
            if i < dump:
                orig = np.array(Image.fromarray(im).resize((res[1], res[0])))
                Image.fromarray(np.concatenate([orig, recon], axis=1)).save(
                    local_out / "recon_samples" / f"{tag}_{i}.png"
                )
        return float(np.mean(per)), pfrr(results, normalized=True)

    train_pfrr, train_by = eval_cards(train_cards, "train", dump=4)
    test_pfrr, test_by = eval_cards(test_cards, "test", dump=4)
    print(f"BIG-FONT PFRR  train={train_pfrr:.3f}  held-out={test_pfrr:.3f}")
    print(f"train by field: {train_by}")
    print(f"test  by field: {test_by}")

    payload = {
        "mode": "bigfont_probe", "font_size": args.font_size, "resolution": args.resolution,
        "n_train": args.n_train, "n_test": args.n_test, "epochs": args.epochs,
        "train_pfrr": train_pfrr, "test_pfrr": test_pfrr,
        "train_by_field": train_by, "test_by_field": test_by, "fingerprint": run_fingerprint(),
    }
    (local_out / "bigfont.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote bigfont.json -> {args.out}")


if __name__ == "__main__":
    main()
