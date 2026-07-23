"""Learned defense vs flat noise (MASTER_PLAN S8) — can an anisotropic transform beat blanket noise?

Trains RedactionProjection P (index-time, per-patch) at a sweep of privacy weights lambda, then measures
the privacy/utility frontier on HELD-OUT cards with HELD-OUT names (open-set) and compares it to the
flat-Gaussian frontier on the same cards. If learned-P dominates flat noise (more utility at matched
privacy), the defense is real and beats the naive baseline — flipping Claim 3 from "impossible" to
"here is the defense and its provable frontier."

  utility = topic-query Recall@1 over the held-out corpus (retrieval preserved?)
  privacy = 1 - name dictionary-attack top-1 (attack suppressed?)
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/learned_defense")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--np-train", type=int, default=128)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--lams", default="0,1,2,5,10,20")
    ap.add_argument("--noise", default="0,0.1,0.2,0.35,0.5,0.75")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.redact import maxsim_batch, train_redactor
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lams = [float(x) for x in args.lams.split(",")]
    noises = [float(x) for x in args.noise.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    rng.shuffle(pool)
    train_names, test_names = pool[:180], pool[180:]  # DISJOINT -> open-set privacy test

    retriever = ColPaliRetriever(model_name=args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for s in pool:  # all names cached (distractors + true)
        q(s)

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=args.font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
            cards.append({"patches": enc.image_patches().astype(np.float32), "name": nm,
                          "topic": topic})
            q(topic)
        return cards

    train = gen(train_names, args.n_train, 1000)
    test = gen(test_names, args.n_test, 5000)

    # --- training tensors (subsample patches to fixed count for stacking/speed) ---
    def sub(p, k):
        idx = rng.choice(p.shape[0], min(k, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c["patches"], args.np_train) for c in train]))
    tr_topic = [torch.tensor(q(c["topic"])) for c in train]
    tr_name = [torch.tensor(q(c["name"])) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]

    # --- eval helpers (torch, on full patches) ---
    def apply_P(P, patches_np):
        with torch.no_grad():
            return P(torch.tensor(patches_np).to(device)) if P is not None else torch.tensor(patches_np).to(device)

    def frontier_point(P=None, sigma=0.0):
        Pt = []
        for c in test:
            p = c["patches"]
            if P is None and sigma > 0:  # flat-noise baseline
                noise = rng.standard_normal(p.shape).astype(np.float32)
                p = p + sigma * np.linalg.norm(p, axis=1, keepdims=True) * noise
                p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
            Pt.append(apply_P(P, p))
        # utility: topic Recall@1 over the test corpus
        hit_u = []
        for i, c in enumerate(test):
            tq = torch.tensor(q(c["topic"])).to(device)
            scores = torch.stack([maxsim_batch(tq, d[None])[0] for d in Pt])
            hit_u.append(int(torch.argmax(scores).item() == i))
        # privacy: 1 - name dictionary attack top-1
        rec = []
        for i, c in enumerate(test):
            cands = [c["name"]] + list(rng.choice([x for x in pool if x != c["name"]], args.distractors, replace=False))
            sc = torch.stack([maxsim_batch(torch.tensor(q(cc)).to(device), Pt[i][None])[0] for cc in cands])
            rec.append(int(torch.argmax(sc).item() == 0))
        return float(np.mean(hit_u)), 1.0 - float(np.mean(rec))

    learned, flat = [], []
    for lam in lams:
        P = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=lam, dim=128,
                           epochs=args.epochs, device=device, seed=args.seed, distractors=distr_q)
        u, p = frontier_point(P=P)
        learned.append({"lambda": lam, "utility": u, "privacy": p})
        print(f"LEARNED lam={lam:>4}: utility={u:.3f} privacy={p:.3f}")
    for sigma in noises:
        u, p = frontier_point(P=None, sigma=sigma)
        flat.append({"noise": sigma, "utility": u, "privacy": p})
        print(f"FLAT   sig={sigma:>4}: utility={u:.3f} privacy={p:.3f}")

    def util_at_priv(pts, target):
        P_ = np.array([x["privacy"] for x in pts]); U = np.array([x["utility"] for x in pts])
        o = np.argsort(P_)
        return float(np.interp(target, P_[o], U[o]))

    dom = {}
    for tp in (0.5, 0.8, 0.9):
        lu, fu = util_at_priv(learned, tp), util_at_priv(flat, tp)
        dom[tp] = {"learned_util": lu, "flat_util": fu, "delta": lu - fu}
    print("DOMINANCE (learned utility - flat utility at matched privacy):",
          {k: round(v["delta"], 3) for k, v in dom.items()})
    verdict = "LEARNED DEFENSE BEATS FLAT NOISE" if all(v["delta"] > 0.05 for v in dom.values()) else "NO CLEAR WIN over flat noise"
    print("VERDICT:", verdict)

    payload = {"mode": "learned_defense", "n_train": args.n_train, "n_test": args.n_test,
               "open_set_names": True, "learned_frontier": learned, "flat_frontier": flat,
               "dominance": dom, "verdict": verdict, "fingerprint": run_fingerprint()}
    (local_out / "learned_defense.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote learned_defense.json -> {args.out}")


if __name__ == "__main__":
    main()
