"""Claim 1 — multi-vector leaks more than pooled (MASTER_PLAN S6, the central novelty).

Runs the verb-validated retrieval attack head-to-head on the SAME cards with two encoders that share
backbone + training data and differ ONLY in aggregation:
  * ColPali  — 1024 per-patch vectors (multi-vector, late interaction)
  * BiPali   — the same patches mean-pooled to ONE page vector (bi-encoder control)

If ColPali recovers field-level PII and BiPali (pooled) does not, the leak is a property of the
MULTI-VECTOR ARCHITECTURE, not of the backbone — i.e. this is not generic CLIP/embedding inversion.
Reports top-1 recovery per field for each, with n + bootstrap 95% CI, a paired McNemar-style delta,
and the per-page STORAGE (floats) so the capacity gap is stated honestly (1a architecture claim; a
matched-bytes 1b control is future work).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _boot_ci(hits, n_res=4000, seed=0):
    a = np.asarray(hits, float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, a.size, (n_res, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/claim1")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=60, help="varied cards")
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.bipali import PooledRetriever
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    def rid():
        return f"{int(rng.integers(10_000_000, 99_999_999))}"

    def rdob():
        return f"{int(rng.integers(1,13)):02d}/{int(rng.integers(1,29)):02d}/{int(rng.integers(1950,2005))}"

    pools = {
        "name": [f"{a} {b}" for a in _FIRST for b in _LAST],
        "id_no": list({rid() for _ in range(args.distractors * 2)})[: args.distractors],
        "dob": list({rdob() for _ in range(args.distractors * 2)})[: args.distractors],
    }

    cards = generate_id_cards(args.n, seed=42, value_font_size=args.font_size, vary=True)
    colpali = ColPaliRetriever(model_name=args.model)
    bipali = PooledRetriever(colpali)

    def run(retriever, tag):
        qc: dict[str, np.ndarray] = {}

        def q(s):
            if s not in qc:
                qc[s] = retriever.encode_query(s)
            return qc[s]

        for pool in pools.values():
            for s in pool:
                q(s)
        hits = {"name": [], "id_no": [], "dob": []}
        # keep per-card, per-field correctness aligned so we can pair ColPali vs BiPali
        keyed = {}
        for ci, (im, fs) in enumerate(cards):
            enc = retriever.encode_page(im)
            bytes_per_page = enc.patches.size  # floats stored
            for f in fs:
                cands = [f.text] + list(rng.choice([x for x in pools[f.field_type] if x != f.text],
                                                   min(args.distractors, len(pools[f.field_type]) - 1), replace=False))
                sc = np.array([maxsim(q(c), enc.patches) for c in cands])
                hit = int(np.argmax(sc) == 0)
                hits[f.field_type].append(hit)
                keyed[(ci, f.field_type)] = hit
        return hits, keyed, bytes_per_page

    cp_hits, cp_keyed, cp_bytes = run(colpali, "colpali")
    bp_hits, bp_keyed, bp_bytes = run(bipali, "bipali")

    summary = {"storage_floats_per_page": {"colpali": cp_bytes, "bipali": bp_bytes,
                                           "ratio": round(cp_bytes / max(bp_bytes, 1), 1)},
               "by_field": {}}
    print(f"storage/page: ColPali {cp_bytes} floats vs BiPali {bp_bytes} floats ({cp_bytes // max(bp_bytes,1)}x)")
    print(f"{'field':8s} | ColPali top1[CI]        | BiPali top1[CI]         | delta | discordant(cp-only/bp-only)")
    for ft in ("name", "id_no", "dob"):
        cm, clo, chi = _boot_ci(cp_hits[ft], seed=1)
        bm, blo, bhi = _boot_ci(bp_hits[ft], seed=2)
        # paired discordant pairs (McNemar): cp right & bp wrong vs bp right & cp wrong
        cp_only = sum(1 for k in cp_keyed if cp_keyed[k] and not bp_keyed[k] and k[1] == ft)
        bp_only = sum(1 for k in bp_keyed if bp_keyed[k] and not cp_keyed[k] and k[1] == ft)
        summary["by_field"][ft] = {
            "colpali": {"top1": cm, "ci": [clo, chi]}, "bipali": {"top1": bm, "ci": [blo, bhi]},
            "delta": cm - bm, "n": len(cp_hits[ft]),
            "discordant_colpali_only": cp_only, "discordant_bipali_only": bp_only,
            "chance": 1.0 / (args.distractors + 1),
        }
        print(f"{ft:8s} | {cm:.3f}[{clo:.3f},{chi:.3f}] | {bm:.3f}[{blo:.3f},{bhi:.3f}] | {cm-bm:+.3f} | {cp_only}/{bp_only}")

    payload = {"mode": "claim1", "n": args.n, "font_size": args.font_size,
               "distractors": args.distractors, "summary": summary, "fingerprint": run_fingerprint()}
    (local_out / "claim1.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote claim1.json -> {args.out}")


if __name__ == "__main__":
    main()
