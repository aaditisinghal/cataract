"""Claim 3 — patch-scoped defense vs the retrieval attack (MASTER_PLAN S8, the defense's real test).

We already showed patch-scoped perturbation preserves retrieval utility; this shows it actually STOPS
the retrieval attack, and dominates flat Gaussian on the privacy/utility frontier.

Setup: each card has PII (name/dob/id) AND a non-PII "issuing office" line = the legitimate retrieval
topic. Two defenses, swept over noise:
  * flat        — perturb ALL image patches
  * patch-scoped— perturb ONLY the PII patches (oracle localizer over name/dob/id boxes; office left alone)

Per (defense, noise):
  * PRIVACY = 1 - retrieval-attack name recovery on the defended encoding (higher = safer)
  * UTILITY = Recall@1 of the office topic-query over the defended corpus (higher = still usable)

Expectation: both defenses reach high privacy (both perturb PII patches), but patch-scoped keeps
UTILITY high (office patches intact) while flat destroys it -> patch-scoped dominates. Bootstrap CIs.
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
    ap.add_argument("--out", default="results/defense_frontier")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--distractors", type=int, default=200)
    ap.add_argument("--font-size", type=int, default=24)
    ap.add_argument("--noise", default="0,0.1,0.2,0.35,0.5,0.75")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from patchguard.data.synthdoc import _FIRST, _LAST, generate_id_cards
    from patchguard.defense.localize import OracleLocalizer
    from patchguard.defense.perturb import flat_gaussian, patch_scoped_gaussian
    from patchguard.repro import run_fingerprint, seed_everything
    from patchguard.retrievers.base import maxsim
    from patchguard.retrievers.colpali import ColPaliRetriever
    from experiments.train_funsd import _gcs_upload

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    noises = [float(x) for x in args.noise.split(",")]
    local_out = Path(tempfile.mkdtemp()) if str(args.out).startswith("gs://") else Path(args.out)
    local_out.mkdir(parents=True, exist_ok=True)

    retriever = ColPaliRetriever(model_name=args.model)
    qcache: dict[str, np.ndarray] = {}

    def q(s):
        if s not in qcache:
            qcache[s] = retriever.encode_query(s)
        return qcache[s]

    name_pool = [f"{a} {b}" for a in _FIRST for b in _LAST]
    for s in name_pool:
        q(s)

    cards = generate_id_cards(args.n, seed=7, value_font_size=args.font_size, vary=True, with_topic=True)
    loc = OracleLocalizer()
    base = []  # (clean_enc, pii_mask, name, topic_query_str, wh)
    for im, fs in cards:
        enc = retriever.encode_page(im)
        wh = (im.shape[1], im.shape[0])
        pii_boxes = [f.box for f in fs if f.field_type in ("name", "dob", "id_no")]
        mask = loc.locate(enc, pii_boxes, wh)
        name = next(f.text for f in fs if f.field_type == "name")
        topic = next((f.text for f in fs if f.field_type == "office"), "ISSUING OFFICE")
        base.append((enc, mask, name, topic, wh))
        q(topic)

    def attack_name(enc, true_name):
        cands = [true_name] + list(rng.choice([x for x in name_pool if x != true_name],
                                              min(args.distractors, len(name_pool) - 1), replace=False))
        sc = np.array([maxsim(q(c), enc.patches) for c in cands])
        return int(np.argmax(sc) == 0)

    def utility_recall1(defended):
        # topic query i should retrieve defended page i as rank 1 over the defended corpus
        hits = []
        allp = [d.patches for d in defended]
        for i, (_, _, _, topic, _) in enumerate(base):
            qi = q(topic)
            scores = np.array([maxsim(qi, p) for p in allp])
            hits.append(int(np.argmax(scores) == i))
        return hits

    frontier = {"flat": [], "patch_scoped": []}
    for sigma in noises:
        for name, defend in (("flat", "flat"), ("patch_scoped", "scoped")):
            if sigma == 0.0:
                defended = [b[0] for b in base]
            elif defend == "flat":
                defended = [flat_gaussian(b[0], sigma, seed=args.seed) for b in base]
            else:
                defended = [patch_scoped_gaussian(b[0], b[1], sigma, seed=args.seed) for b in base]
            recov = [attack_name(d, base[i][2]) for i, d in enumerate(defended)]
            util = utility_recall1(defended)
            pm, plo, phi = _ci([1 - r for r in recov], seed=1)
            um, ulo, uhi = _ci(util, seed=2)
            frontier[name].append({"noise": sigma, "privacy": pm, "privacy_ci": [plo, phi],
                                   "utility": um, "utility_ci": [ulo, uhi]})
        f = frontier["flat"][-1]
        s = frontier["patch_scoped"][-1]
        print(f"noise {sigma:>4}: FLAT priv={f['privacy']:.2f} util={f['utility']:.2f} | "
              f"SCOPED priv={s['privacy']:.2f} util={s['utility']:.2f}")

    # dominance: at matched privacy, patch-scoped utility - flat utility (interp on privacy)
    def util_at_priv(pts, target):
        p = np.array([x["privacy"] for x in pts]); u = np.array([x["utility"] for x in pts])
        o = np.argsort(p)
        return float(np.interp(target, p[o], u[o]))

    dom = {}
    for tp in (0.8, 0.9, 0.95):
        us = util_at_priv(frontier["patch_scoped"], tp)
        uf = util_at_priv(frontier["flat"], tp)
        dom[tp] = {"scoped_util": us, "flat_util": uf, "delta": us - uf}
    print("dominance (utility at matched privacy):", {k: round(v["delta"], 3) for k, v in dom.items()})

    payload = {"mode": "defense_frontier", "n": args.n, "noises": noises,
               "frontier": frontier, "dominance": dom, "fingerprint": run_fingerprint()}
    (local_out / "defense_frontier.json").write_text(json.dumps(payload, indent=2))
    if str(args.out).startswith("gs://"):
        _gcs_upload(local_out, args.out)
    print(f"wrote defense_frontier.json -> {args.out}")


if __name__ == "__main__":
    main()
