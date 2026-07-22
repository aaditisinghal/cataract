"""The ★ kill test on REAL data (MASTER_PLAN gate #2, real path).

Trains a decoder for ColPali AND for pooled-BiPali, then on held-out FUNSD measures:
  * PFRR(ColPali) vs PFRR(BiPali)                       -> the architecture delta (Claim 1)
  * frontier: flat vs oracle-patch-scoped Gaussian      -> patch-scoped dominance (Claim 3)
and applies the pre-registered gate (>=15pp delta AND patch-scoped frontier CI excludes 0).

Utility is per-doc reciprocal rank in self-retrieval over the defended test corpus; privacy is
1 - per-doc PFRR. Parametrized small for a first real signal; scale up (--test-limit 200, 5 seeds)
for the paper number.

Runs in the repro container (has ColPali + tesseract). GCS-aware --data/--out like train_funsd.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from patchguard.attack.train import Sample, TrainConfig, build_dataset, train_decoder
from patchguard.data.fields import AnnotatedField
from patchguard.defense.localize import OracleLocalizer
from patchguard.defense.perturb import flat_gaussian, patch_scoped_gaussian
from patchguard.eval.killgate import assemble_and_gate
from patchguard.eval.reconstruct import TesseractOCR, ocr_field_pfrr, reconstruct
from patchguard.repro import run_fingerprint, seed_everything


@dataclass
class Page:
    """A test page carrying the loaded image AND full field annotations (text needed for PFRR)."""

    image: np.ndarray
    fields: list[AnnotatedField]
    size: tuple[int, int]  # (width, height)

    def boxes(self) -> list[tuple[float, float, float, float]]:
        return [f.box for f in self.fields]


def load_pages(root: str, split: str, limit: int | None) -> list[Page]:
    from PIL import Image

    from patchguard.data.funsd import iter_funsd

    pages: list[Page] = []
    for ps in iter_funsd(root, split=split, granularity="word"):
        img = np.array(Image.open(ps.image_path).convert("RGB"))
        pages.append(Page(image=img, fields=ps.fields, size=ps.size))
        if limit and len(pages) >= limit:
            break
    if not pages:
        raise RuntimeError(f"no pages under {root}/{split}")
    return pages


def _doc_query_text(s: Page, max_len: int = 200) -> str:
    return " ".join(f.text for f in s.fields)[:max_len] or "document"


def _per_doc_recovery(decoder, retriever, samples, encs, ocr, out_size, device, max_fields):
    """Per-doc PFRR (fraction of that doc's fields recovered) on given (possibly defended) encodings."""
    rates = np.zeros(len(samples))
    for i, (s, enc) in enumerate(zip(samples, encs)):
        recon = reconstruct(decoder, enc, device)
        fields = s.fields[:max_fields]
        res = ocr_field_pfrr(recon, fields, s.size, out_size, enc.resize_policy, ocr)
        rates[i] = sum(r.normalized_exact for r in res) / max(len(res), 1)
    return rates


def _reciprocal_ranks(retriever, samples, page_encs):
    """Per-doc self-retrieval reciprocal rank over the (defended) corpus = utility proxy."""
    queries = [retriever.encode_query(_doc_query_text(s)) for s in samples]
    rr = np.zeros(len(samples))
    for i, q in enumerate(queries):
        scores = np.array([retriever.score(q, penc) for penc in page_encs])
        rank = 1 + int((scores > scores[i]).sum())  # 1 = best
        rr[i] = 1.0 / rank
    return rr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="FUNSD root (local or gs://)")
    ap.add_argument("--out", default="results/killtest")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-limit", type=int, default=120)
    ap.add_argument("--test-limit", type=int, default=20)
    ap.add_argument("--max-fields", type=int, default=25)
    ap.add_argument("--noise-levels", type=int, default=5)
    ap.add_argument("--max-noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    import torch

    from patchguard.retrievers.bipali import PooledRetriever
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_download, _gcs_upload

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_size = (448, 448)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    data_root = args.data
    if str(data_root).startswith("gs://"):
        data_root = str(_gcs_download(data_root, Path(tempfile.mkdtemp())))
    # training uses lightweight Samples (only need boxes for the weight map); eval uses full Pages
    train_pages = load_pages(data_root, "training_data", args.train_limit)
    train_s = [Sample(image=p.image, field_boxes=p.boxes(), orig_size=p.size) for p in train_pages]
    test_s = load_pages(data_root, "testing_data", args.test_limit)
    print(f"train={len(train_s)} test={len(test_s)}")

    colpali = ColPaliRetriever(model_name=args.model)
    bipali = PooledRetriever(colpali)
    ocr = TesseractOCR()
    noises = list(np.linspace(0.0, args.max_noise, args.noise_levels))

    decoders = {}
    for name, retr in (("colpali", colpali), ("bipali", bipali)):
        cfg = TrainConfig(out_size=out_size, epochs=args.epochs, seed=args.seed,
                          ckpt_dir=str(local_out / f"decoder_{name}"), use_lpips=False, w_lpips=0.0)
        ds = build_dataset(train_s, retr, cfg)
        train_decoder(ds, cfg)
        # rebuild the trained module in memory
        from patchguard.attack.decoder import PatchGridDecoder
        dec = PatchGridDecoder(dim=ds.dim, grid=ds.grid, out_size=out_size).to(device)
        dec.load_state_dict(torch.load(local_out / f"decoder_{name}" / "decoder.pt")["state_dict"])
        decoders[name] = dec
        print(f"trained {name} decoder (grid={ds.grid})")

    # undefended per-doc PFRR for the architecture delta
    test_encs_cp = [colpali.encode_page(s.image) for s in test_s]
    test_encs_bp = [bipali.encode_page(s.image) for s in test_s]
    rec_cp = _per_doc_recovery(decoders["colpali"], colpali, test_s, test_encs_cp, ocr, out_size, device, args.max_fields)
    rec_bp = _per_doc_recovery(decoders["bipali"], bipali, test_s, test_encs_bp, ocr, out_size, device, args.max_fields)
    print(f"undefended PFRR: colpali={rec_cp.mean():.3f} bipali={rec_bp.mean():.3f}")

    # frontier: flat vs oracle-patch-scoped on ColPali, per-doc priv (1-PFRR) + util (reciprocal rank)
    loc = OracleLocalizer()
    masks = [loc.locate(e, s.boxes(), s.size) for e, s in zip(test_encs_cp, test_s)]
    nD, nN = len(test_s), len(noises)
    priv_patch = np.zeros((nD, nN)); util_patch = np.zeros((nD, nN))
    priv_flat = np.zeros((nD, nN)); util_flat = np.zeros((nD, nN))
    for j, sigma in enumerate(noises):
        patch_encs = [patch_scoped_gaussian(e, m, sigma, seed=args.seed) for e, m in zip(test_encs_cp, masks)]
        flat_encs = [flat_gaussian(e, sigma, seed=args.seed) for e in test_encs_cp]
        priv_patch[:, j] = 1.0 - _per_doc_recovery(decoders["colpali"], colpali, test_s, patch_encs, ocr, out_size, device, args.max_fields)
        priv_flat[:, j] = 1.0 - _per_doc_recovery(decoders["colpali"], colpali, test_s, flat_encs, ocr, out_size, device, args.max_fields)
        util_patch[:, j] = _reciprocal_ranks(colpali, test_s, patch_encs)
        util_flat[:, j] = _reciprocal_ranks(colpali, test_s, flat_encs)
        print(f"noise {sigma:.2f}: priv_patch={priv_patch[:,j].mean():.3f} priv_flat={priv_flat[:,j].mean():.3f} "
              f"util_patch={util_patch[:,j].mean():.3f} util_flat={util_flat[:,j].mean():.3f}")

    result = assemble_and_gate(rec_cp, rec_bp, util_patch, priv_patch, util_flat, priv_flat,
                               min_delta_pp=15.0, n_resamples=2000, seed=args.seed)
    print("\n" + result.summary())

    payload = {
        "mode": "real", "n_test": nD, "noises": [float(x) for x in noises],
        "pfrr_colpali": result.pfrr_colpali, "pfrr_bipali": result.pfrr_bipali,
        "pfrr_delta_pp": result.pfrr_delta_pp, "delta_pass": result.delta_pass,
        "auc_diff": result.auc_diff, "auc_ci": list(result.auc_ci),
        "frontier_pass": result.frontier_pass, "decision": result.decision,
        "config": vars(args), "fingerprint": run_fingerprint(),
    }
    (local_out / "killtest.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote killtest.json -> {args.out}")


if __name__ == "__main__":
    main()
