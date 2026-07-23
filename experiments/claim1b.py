"""Claim 1b — matched-bytes control (MASTER_PLAN S6). Kills the "ColPali just has 1030x more floats".

If ColPali's advantage were pure capacity, then at MATCHED storage it should match BiPali. We subsample
ColPali to K random patches (K*128 floats) and sweep K; BiPali is a single pooled 128-float vector. The
key comparison is K=1 ColPali (128 floats, matched to BiPali) vs BiPali (128 floats). Because the leak
is holographic (every ColPali patch is a global summary), even 1 late-interaction patch should recover
what BiPali's mean-pooled 128-vector cannot — i.e. the advantage is the AGGREGATION (late interaction vs
mean pool), not the byte budget.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _ci(hits, n_res=3000, seed=0):
    a = np.asarray(hits, float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = a[rng.integers(0, a.size, (n_res, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/claim1b")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--ks", default="1,2,4,8,16,64,256")
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
    ks = [int(x) for x in args.ks.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    colpali = ColPaliRetriever(model_name=args.model)
    bipali = PooledRetriever(colpali)
    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]

    cpq: dict[str, np.ndarray] = {}
    bpq: dict[str, np.ndarray] = {}

    def qc(s):
        if s not in cpq:
            cpq[s] = colpali.encode_query(s)
        return cpq[s]

    def qb(s):
        if s not in bpq:
            bpq[s] = bipali.encode_query(s)
        return bpq[s]

    for s in name_pool:
        qc(s); qb(s)

    def attack(patches, true_name, qfn):
        cands = [true_name] + list(rng.choice([x for x in name_pool if x != true_name],
                                              min(args.distractors, len(name_pool) - 1), replace=False))
        sc = np.array([maxsim(qfn(c), patches) for c in cands])
        return int(np.argmax(sc) == 0)

    cards = generate_id_cards(args.n, seed=11, value_font_size=args.font_size, vary=True)
    cp_img, bp_vec, names = [], [], []
    for im, fs in cards:
        cp_img.append(colpali.encode_page(im).image_patches())  # (1024, 128)
        bp_vec.append(bipali.encode_page(im).patches)           # (1, 128)
        names.append(next(f.text for f in fs if f.field_type == "name"))

    sweep = []
    for K in ks:
        rec = []
        for i in range(len(cards)):
            idx = rng.choice(cp_img[i].shape[0], min(K, cp_img[i].shape[0]), replace=False)
            rec.append(attack(cp_img[i][idx], names[i], qc))
        m, lo, hi = _ci(rec, seed=K)
        sweep.append({"k_patches": K, "floats": K * 128, "recovery": m, "ci": [lo, hi]})
        print(f"ColPali K={K:>4} ({K*128:>6} floats): recovery={m:.3f} [{lo:.3f},{hi:.3f}]")

    bp = _ci([attack(bp_vec[i], names[i], qb) for i in range(len(cards))], seed=99)
    print(f"BiPali  pooled (   128 floats): recovery={bp[0]:.3f} [{bp[1]:.3f},{bp[2]:.3f}]  (chance {1/(args.distractors+1):.4f})")
    matched = {"colpali_k1_128floats": sweep[0]["recovery"], "bipali_128floats": bp[0],
               "delta_at_matched_bytes": sweep[0]["recovery"] - bp[0]}
    print(f"MATCHED-BYTES (128 floats): ColPali-1patch {matched['colpali_k1_128floats']:.3f} vs "
          f"BiPali {matched['bipali_128floats']:.3f}  delta={matched['delta_at_matched_bytes']:+.3f}")

    payload = {"mode": "claim1b", "n": len(cards), "sweep": sweep,
               "bipali": {"recovery": bp[0], "ci": [bp[1], bp[2]], "floats": 128},
               "matched_bytes": matched, "chance": 1 / (args.distractors + 1),
               "fingerprint": run_fingerprint()}
    (local_out / "claim1b.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote claim1b.json -> {args.out}")


if __name__ == "__main__":
    main()
