"""The constructive answer to §4.13: an information-DESTROYING defense that survives the inverse attack.

§4.13 broke the residual ``RedactionProjection``: it is (near-)bijective, so the B1b inverse attacker
learns ``P^-1`` (recon cosine ~0.998) and recovers PII at 1.00. This experiment tests whether a
RANK-DEFICIENT defense — ``NullspaceRedaction``, which annihilates the name-discriminative subspace of
every stored patch — defeats that SAME adaptive attacker, at what utility cost.

For a sweep over removed-rank ``k`` we report, on held-out open-set cards:
  * non_adaptive   : vanilla dictionary attack recovery on P_null(patch)          (should fall with k)
  * inverse (B1b)  : the attack that BROKE the residual P, run against P_null      (should ALSO fall —
                     the info is gone, not hidden, so P^-1 cannot recover it)
  * utility        : topic/office retrieval Recall@1 on the P_null index          (should hold until k
                     starts eating content directions)
A single reference point trains the residual ``RedactionProjection`` and runs the inverse attack on it
(reproducing ~1.00) so the two defenses are compared in-run: the residual P hides, ``NullspaceRedaction``
removes. A defensible defense is the ``k`` where the INVERSE recovery collapses while utility holds.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Information-destroying (nullspace) defense vs the inverse attack.")
    ap.add_argument("--out", default="results/certified_defense")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n-train", type=int, default=64, help="cards for building directions / P / inverse pairs")
    ap.add_argument("--n-test", type=int, default=40, help="held-out open-set victim/eval cards")
    ap.add_argument("--distractors", type=int, default=200, help="dictionary lineup size")
    ap.add_argument("--ks", default="0,2,4,8,16,32", help="removed-rank sweep for NullspaceRedaction")
    ap.add_argument("--r-topic", type=int, default=8, help="topic directions spared before extracting PII dirs")
    ap.add_argument("--inv-epochs", type=int, default=400, help="inverse-net Pinv training epochs")
    ap.add_argument("--lam", type=float, default=5.0, help="lambda for the residual-P reference point")
    ap.add_argument("--p-epochs", type=int, default=300, help="residual-P training epochs (reference)")
    ap.add_argument("--np-train", type=int, default=128)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from experiments.adaptive_attack import Card, baseline_dictionary, strat_inverse, _apply_P
    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_card
    from patchguard.defense.nullspace import NullspaceRedaction, pii_directions
    from patchguard.defense.redact import train_redactor
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(x) for x in str(args.ks).split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    pool = [f"{a} {b}" for a in _FIRST for b in _LAST]  # closed vocabulary K=240
    name_to_idx = {nm: i for i, nm in enumerate(pool)}
    pool_shuf = list(pool)
    rng.shuffle(pool_shuf)
    train_names, test_names = pool_shuf[:180], pool_shuf[180:]  # DISJOINT -> open-set

    from experiments.adaptive_attack import _build_retriever

    retriever = _build_retriever(args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    for s in pool:
        q(s)

    def gen(names, k, seed0):
        cards = []
        for i in range(k):
            nm = names[int(rng.integers(0, len(names)))]
            im, fs = generate_id_card(seed0 + i, value_font_size=args.font_size, vary=True,
                                      fixed_name=nm, with_topic=True)
            enc = retriever.encode_page(im)
            topic = next((f.text for f in fs if f.field_type == "office"), "OFFICE")
            cards.append(Card(patches=enc.image_patches().astype(np.float32), name=nm,
                              name_idx=name_to_idx[nm], topic=topic))
            q(topic)
        return cards

    train = gen(train_names, args.n_train, 1000)   # direction estimation + P + inverse pairs
    test = gen(test_names, args.n_test, 5000)       # held-out open-set victims
    topic_pool = sorted({c.topic for c in (train + test)})
    for t in topic_pool:
        q(t)

    # ---- PII-discriminative directions from TRAIN queries (attacker-agnostic; uses public encoder) ----
    name_tokens = np.concatenate([q(n) for n in train_names], axis=0)   # (T_name, d)
    topic_tokens = np.concatenate([q(t) for t in topic_pool], axis=0)     # (T_topic, d)

    def topic_recall(P):
        Pt = [_apply_P(P, c.patches, device).cpu().numpy() for c in test]
        hits = []
        for i, c in enumerate(test):
            sc = [maxsim(np.asarray(q(c.topic), np.float32), Pt[j]) for j in range(len(test))]
            hits.append(int(np.argmax(sc) == i))
        return float(np.mean(hits)) if hits else 0.0

    dict_chance = 1.0 / (min(args.distractors, len(pool) - 1) + 1)
    undefended = baseline_dictionary(test, None, q, pool, args.distractors, device, rng)
    util_clean = topic_recall(None)
    print(f"ANCHOR undefended dict={undefended:.3f} | clean topic Recall@1={util_clean:.3f} "
          f"(chance {dict_chance:.4f})")

    # ---- sweep the nullspace rank k ----
    sweep = []
    for k in ks:
        D = pii_directions(name_tokens, topic_tokens, k=k, r_topic=args.r_topic).to(device)
        Pnull = NullspaceRedaction(D).to(device).eval()
        non_adapt = baseline_dictionary(test, Pnull, q, pool, args.distractors, device, rng)
        inv_rec, inv_m = strat_inverse(train, test, Pnull, q, pool, args.distractors, device,
                                       args.inv_epochs, args.seed, rng)
        util = topic_recall(Pnull)
        row = {"k": k, "non_adaptive_recovery": non_adapt, "non_adaptive_privacy": 1.0 - non_adapt,
               "inverse_recovery": inv_rec, "inverse_privacy": 1.0 - inv_rec,
               "utility_topic_recall": util, "recon_cosine": inv_m.get("recon_cosine")}
        sweep.append(row)
        print(f"k={k:>3} | non-adapt rec={non_adapt:.3f} (priv {1-non_adapt:.3f}) | "
              f"INVERSE rec={inv_rec:.3f} (priv {1-inv_rec:.3f}) | util={util:.3f} | "
              f"recon_cos={inv_m.get('recon_cosine'):.3f}")

    # ---- reference: residual RedactionProjection under the SAME inverse attack (reproduce §4.13 ~1.0) ----
    def sub(p, kk):
        idx = rng.choice(p.shape[0], min(kk, p.shape[0]), replace=False)
        return p[idx]

    tr_patches = torch.tensor(np.stack([sub(c.patches, args.np_train) for c in train]))
    tr_topic = [torch.tensor(q(c.topic)) for c in train]
    tr_name = [torch.tensor(q(c.name)) for c in train]
    distr_q = [torch.tensor(q(n)) for n in list(rng.choice(train_names, 16, replace=False))]
    Pres = train_redactor(tr_patches, tr_topic, tr_name, distr_q, lam=args.lam, dim=128,
                          epochs=args.p_epochs, device=device, seed=args.seed, distractors=distr_q)
    res_non_adapt = baseline_dictionary(test, Pres, q, pool, args.distractors, device, rng)
    res_inv, res_m = strat_inverse(train, test, Pres, q, pool, args.distractors, device,
                                   args.inv_epochs, args.seed, rng)
    res_util = topic_recall(Pres)
    residual_ref = {"defense": "RedactionProjection(residual)", "lam": args.lam,
                    "non_adaptive_recovery": res_non_adapt, "inverse_recovery": res_inv,
                    "utility_topic_recall": res_util, "recon_cosine": res_m.get("recon_cosine")}
    print(f"REF residual P (lam={args.lam}): non-adapt rec={res_non_adapt:.3f} | INVERSE rec={res_inv:.3f} "
          f"(recon_cos={res_m.get('recon_cosine'):.3f}) | util={res_util:.3f}")

    # ---- verdict: best k where the INVERSE attack is defeated while utility holds ----
    safe = [r for r in sweep if r["inverse_recovery"] <= 3.0 * dict_chance and r["utility_topic_recall"] >= 0.8]
    if safe:
        best = max(safe, key=lambda r: r["utility_topic_recall"])
        verdict = (f"CONSTRUCTIVE: NullspaceRedaction at k={best['k']} defeats the INVERSE attacker "
                   f"(inverse recovery {best['inverse_recovery']:.3f} vs {res_inv:.3f} for the residual P) "
                   f"while keeping topic utility {best['utility_topic_recall']:.3f}. Removing the PII "
                   f"subspace destroys the information the residual transform only hid.")
    else:
        verdict = ("NO clean k defeats the inverse attack while holding utility>=0.8 in this run — the PII "
                   "and content subspaces overlap here; report the frontier and the fundamental floor.")
    print("VERDICT:", verdict)

    payload = {"mode": "certified_defense", "model": args.model, "n_train": args.n_train,
               "n_test": args.n_test, "distractors": args.distractors, "dict_chance": dict_chance,
               "r_topic": args.r_topic, "undefended_dict": undefended, "utility_clean": util_clean,
               "sweep": sweep, "residual_reference": residual_ref, "verdict": verdict,
               "fingerprint": run_fingerprint()}
    (local_out / "certified_defense.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"\nwrote certified_defense.json -> {args.out}")


if __name__ == "__main__":
    main()
