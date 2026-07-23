"""Cross-model generalization (MASTER_PLAN S2/S6/S10) — is the leak + bleed a ColPali quirk?

Runs the model-agnostic core tests on ANY multi-vector retriever (ColPali or ColQwen2), needing only
raw patches (no box alignment):
  * LEAK       : retrieval-attack name recovery (does it leak field PII?)
  * WRONG-PAGE : name_i vs page_i (correct) vs page_j (wrong) — is the attack genuine?
  * BLEED      : remove a RANDOM fraction of patches, re-attack — if recovery survives large removals,
                 the info is holographically distributed (model-agnostic version of the erasure result).

If ColQwen2 reproduces ColPali's pattern, the finding generalizes to multi-vector visual retrievers.
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
    ap.add_argument("--out", default="results/cross_model")
    ap.add_argument("--retriever", default="colqwen2", choices=["colpali", "colqwen2"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    if args.retriever == "colqwen2":
        from patchguard.retrievers.colqwen2 import ColQwen2Retriever
        retriever = ColQwen2Retriever(model_name=args.model or "vidore/colqwen2-v1.0")
    else:
        from patchguard.retrievers.colpali import ColPaliRetriever
        retriever = ColPaliRetriever(model_name=args.model or "vidore/colpali-v1.3")

    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    for s in name_pool:
        q(s)

    def attack(patches, true_name):
        cands = [true_name] + list(rng.choice([x for x in name_pool if x != true_name],
                                              min(args.distractors, len(name_pool) - 1), replace=False))
        sc = np.array([maxsim(q(c), patches) for c in cands])
        return int(np.argmax(sc) == 0)

    cards = generate_id_cards(args.n, seed=11, value_font_size=args.font_size, vary=True)
    encs, names = [], []
    for im, fs in cards:
        encs.append(retriever.encode_page(im).patches)
        names.append(next(f.text for f in fs if f.field_type == "name"))
    print(f"{args.retriever}: {len(encs)} pages, patches/page ~ {encs[0].shape}")

    N = len(encs)
    leak = [attack(encs[i], names[i]) for i in range(N)]
    wrong = [attack(encs[(i + 1) % N], names[i]) for i in range(N)]
    bleed = {}
    for frac in (0.25, 0.5, 0.75, 0.9):
        rec = []
        for i in range(N):
            p = encs[i]
            keep = rng.choice(p.shape[0], max(1, int(p.shape[0] * (1 - frac))), replace=False)
            rec.append(attack(p[keep], names[i]))
        bleed[frac] = _ci(rec, seed=int(frac * 100))

    lk, wr = _ci(leak, seed=1), _ci(wrong, seed=2)
    print(f"LEAK (correct)    : {lk[0]:.3f} [{lk[1]:.3f},{lk[2]:.3f}]")
    print(f"WRONG-PAGE        : {wr[0]:.3f} [{wr[1]:.3f},{wr[2]:.3f}]  (chance {1/(args.distractors+1):.4f})")
    for f, (m, lo, hi) in bleed.items():
        print(f"BLEED remove {int(f*100)}% : {m:.3f} [{lo:.3f},{hi:.3f}]")
    generalizes = lk[0] >= 0.8 and wr[0] <= 0.15 and bleed[0.5][0] >= 0.5

    payload = {"mode": "cross_model", "retriever": args.retriever, "n": N,
               "patches_per_page": list(encs[0].shape), "chance": 1 / (args.distractors + 1),
               "leak": lk[0], "wrong_page": wr[0],
               "bleed": {str(f): v[0] for f, v in bleed.items()},
               "cis": {"leak": list(lk[1:]), "wrong_page": list(wr[1:])},
               "generalizes": generalizes, "fingerprint": run_fingerprint()}
    (local_out / "cross_model.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"GENERALIZES: {generalizes}\nwrote cross_model.json -> {args.out}")


if __name__ == "__main__":
    main()
