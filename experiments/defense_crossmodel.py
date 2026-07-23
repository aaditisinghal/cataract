"""Does the learned defense GENERALIZE across multi-vector VLMs? (MASTER_PLAN S8, plan item B4)

The attack's headline is cross-model: the PII leak is a property of multi-vector visual retrieval, not a
ColPali quirk (see experiments/cross_model.py — ColQwen2 reproduces the leak + holographic bleed). A
defense that only worked on ColPali would therefore be a footnote. This experiment runs the SAME learned
index-time redactor (RedactionProjection, trained min-max) on a SELECTABLE backbone and traces its
privacy/utility frontier, mirroring the attack's cross-model result on the defense side.

  utility = topic-query Recall@1 over a held-out corpus (retrieval still works after redaction?)
  privacy = 1 - name dictionary-attack top-1 on the redacted patches (attack suppressed?)

Names are OPEN-SET (train/test name pools are disjoint), so privacy is measured on names P never saw.
A positive result: on ColQwen2 too, sweeping lambda buys real privacy while retaining topic utility ->
the defense is a general multi-vector-VLM primitive, not a single-model artifact. Because ColQwen2 emits a
variable-length patch set with no fixed grid, we operate on the raw ``.patches`` array (model-agnostic),
exactly as the cross-model attack does.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/defense_crossmodel")
    ap.add_argument("--retriever", default="colqwen2", choices=["colpali", "colqwen2"])
    ap.add_argument("--model", default=None, help="override backbone checkpoint (else backend default)")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--distractors", type=int, default=200, help="name lineup size for the privacy attack")
    ap.add_argument("--np-train", type=int, default=128, help="patches subsampled per train doc (for stacking)")
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--lams", default="0,2,5,10")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import maxsim_batch, train_redactor
    from patchguard.repro import run_fingerprint, seed_everything
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lams = [float(x) for x in args.lams.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    if args.retriever == "colqwen2":
        from patchguard.retrievers.colqwen2 import ColQwen2Retriever
        model_name = args.model or "vidore/colqwen2-v1.0"
        retriever = ColQwen2Retriever(model_name=model_name)
    else:
        from patchguard.retrievers.colpali import ColPaliRetriever
        model_name = args.model or "vidore/colpali-v1.3"
        retriever = ColPaliRetriever(model_name=model_name)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # K=240
    rng.shuffle(pool)
    train_names, test_names = pool[:180], pool[180:]  # DISJOINT -> open-set privacy test

    qcache: dict[str, np.ndarray] = {}

    def q(s: str) -> np.ndarray:
        if s not in qcache:
            qcache[s] = np.asarray(retriever.encode_query(s), dtype=np.float32)
        return qcache[s]

    for s in pool:  # warm the full name candidate space
        q(s)

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=args.font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
            cards.append({"patches": np.asarray(enc.patches, dtype=np.float32), "name": nm, "topic": topic})
            q(topic)
        return cards

    train = gen(train_names, args.n_train, 1000)
    test = gen(test_names, args.n_test, 5000)
    dim = int(train[0]["patches"].shape[-1])

    # --- training tensors (subsample patches to a fixed count for stacking; clamp to the corpus min) ---
    npt = min(args.np_train, min(int(c["patches"].shape[0]) for c in train))

    def sub(p, k):
        idx = rng.choice(p.shape[0], min(k, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], npt) for c in train]))
    tr_topic = [torch.tensor(q(c["topic"])) for c in train]
    tr_name = [torch.tensor(q(c["name"])) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]

    # --- eval helpers (torch, on FULL variable-length patches) ---
    def apply_P(P, patches_np):
        with torch.no_grad():
            t = torch.tensor(patches_np).to(device)
            return P(t) if P is not None else t

    def frontier_point(P=None):
        Pt = [apply_P(P, c["patches"]) for c in test]
        # utility: topic Recall@1 over the redacted test corpus
        hit_u = []
        for i, c in enumerate(test):
            tq = torch.tensor(q(c["topic"])).to(device)
            scores = torch.stack([maxsim_batch(tq, d[None])[0] for d in Pt])
            hit_u.append(int(torch.argmax(scores).item() == i))
        # privacy: 1 - name dictionary attack top-1
        rec = []
        for i, c in enumerate(test):
            cands = [c["name"]] + list(rng.choice([x for x in pool if x != c["name"]],
                                                  min(args.distractors, len(pool) - 1), replace=False))
            sc = torch.stack([maxsim_batch(torch.tensor(q(cc)).to(device), Pt[i][None])[0] for cc in cands])
            rec.append(int(torch.argmax(sc).item() == 0))
        return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))

    print(f"{args.retriever} ({model_name}): {len(train)} train / {len(test)} test docs, "
          f"dim={dim}, patches/train-doc subsampled to {npt}")

    frontier = []
    for lam in lams:
        P = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=lam, dim=dim,
                           epochs=args.epochs, device=device, seed=args.seed, distractors=distr_q)
        u, p = frontier_point(P=P)
        frontier.append({"lambda": lam, "utility": u, "privacy": p})
        print(f"  lam={lam:>5}: utility={u:.3f}  privacy={p:.3f}")

    def util_at_priv(pts, target):
        P_ = np.array([x["privacy"] for x in pts]); U = np.array([x["utility"] for x in pts])
        o = np.argsort(P_)
        return float(np.interp(target, P_[o], U[o]))

    util_at = {tp: util_at_priv(frontier, tp) for tp in (0.5, 0.8, 0.9)}
    # generalizes: some lambda reaches strong privacy while keeping retrieval usable on THIS backbone.
    generalizes = any(x["privacy"] >= 0.8 and x["utility"] >= 0.5 for x in frontier)
    print("UTILITY @ PRIVACY:", {k: round(v, 3) for k, v in util_at.items()})
    print("GENERALIZES (priv>=0.8 & util>=0.5 somewhere):", generalizes)

    payload = {"mode": "defense_crossmodel", "retriever": args.retriever, "model": model_name,
               "n_train": args.n_train, "n_test": args.n_test, "open_set_names": True, "dim": dim,
               "patches_per_train_doc_sub": npt, "distractors": args.distractors,
               "frontier": frontier, "utility_at_privacy": {str(k): v for k, v in util_at.items()},
               "generalizes": generalizes, "fingerprint": run_fingerprint()}
    (local_out / "defense_crossmodel.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote defense_crossmodel.json -> {args.out}")


if __name__ == "__main__":
    main()
